import time
import os

# AES parameters
Nb = 4  # Number of columns comprising the State. For AES, Nb = 4
Nk = 4  # Number of 32-bit words comprising the Cipher Key. For AES-128, Nk = 4
Nr = 10  # Number of rounds. For AES-128, Nr = 10

# AES constants
Rcon = [
    0x00,
    0x01,
    0x02,
    0x04,
    0x08,
    0x10,
    0x20,
    0x40,
    0x80,
    0x1B,
    0x36,
]

def nibble_swap(byte):
    """Swap the higher nibble with the lower nibble of a byte."""
    high_nibble = (byte & 0xF0) >> 4
    low_nibble = (byte & 0x0F) << 4
    return high_nibble | low_nibble

def sub_bytes(state):
    """Apply the nibble swap substitution to each byte in the state."""
    for i in range(4):
        for j in range(4):
            state[i][j] = nibble_swap(state[i][j])
    return state

def inv_sub_bytes(state):
    """Inverse nibble swap is the same operation as nibble swap."""
    return sub_bytes(state)

def shift_rows(state):
    """Shift the rows of the state to the left."""
    # Row 0 is not shifted
    state[1] = state[1][1:] + state[1][:1]  # Shift row 1 left by 1
    state[2] = state[2][2:] + state[2][:2]  # Shift row 2 left by 2
    state[3] = state[3][3:] + state[3][:3]  # Shift row 3 left by 3
    return state

def inv_shift_rows(state):
    """Shift the rows of the state to the right."""
    # Row 0 is not shifted
    state[1] = state[1][-1:] + state[1][:-1]   # Shift row 1 right by 1
    state[2] = state[2][-2:] + state[2][:-2]   # Shift row 2 right by 2
    state[3] = state[3][-3:] + state[3][:-3]   # Shift row 3 right by 3
    return state


def xtime(a):
    """Multiply by x (i.e., {02}) in GF(2^8)."""
    return (((a << 1) & 0xFF) ^ 0x1B) if (a & 0x80) else (a << 1)

def mix_single_column(a):
    """Mix one column for the MixColumns step."""
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)
    return a

def mix_columns(state):
    """Mix the columns of the state."""
    
        # Debug: Print state before mixing
    print("State before mix_columns:", state)

    for i in range(4):
        col = [state[j][i] for j in range(4)]
        col = mix_single_column(col)
        for j in range(4):
            state[j][i] = col[j]
    return state

def inv_mix_columns(state):
    """Inverse MixColumns step."""
    for i in range(4):
        s = [state[j][i] for j in range(4)]
        state[0][i] = multiply(s[0], 14) ^ multiply(s[1], 11) ^ multiply(s[2], 13) ^ multiply(s[3], 9)
        state[1][i] = multiply(s[0], 9) ^ multiply(s[1], 14) ^ multiply(s[2], 11) ^ multiply(s[3], 13)
        state[2][i] = multiply(s[0], 13) ^ multiply(s[1], 9) ^ multiply(s[2], 14) ^ multiply(s[3], 11)
        state[3][i] = multiply(s[0], 11) ^ multiply(s[1], 13) ^ multiply(s[2], 9) ^ multiply(s[3], 14)
    return state

def multiply(a, b):
    """Multiply two numbers in GF(2^8) field."""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        high_bit_set = a & 0x80
        a <<= 1
        a &= 0xFF
        if high_bit_set:
            a ^= 0x1B
        b >>= 1
    return p

def add_round_key(state, round_key_words):
    """Add (XOR) the round key to the state."""
    for c in range(4):
        for r in range(4):
            state[r][c] ^= (round_key_words[c] >> (8 * (3 - r))) & 0xFF
    return state

def key_expansion(key):
    """Generate the expanded key from the cipher key."""
    key_symbols = [k for k in key]
    if len(key_symbols) < 4 * Nk:
        key_symbols += [0x01] * (4 * Nk - len(key_symbols))
    key_schedule = []
    for i in range(Nk):
        word = (key_symbols[4*i] << 24) | (key_symbols[4*i+1] << 16) | (key_symbols[4*i+2] << 8) | key_symbols[4*i+3]
        key_schedule.append(word)
    for i in range(Nk, Nb * (Nr + 1)):
        temp = key_schedule[i - 1]
        if i % Nk == 0:
            temp = ((nibble_swap((temp >> 16) & 0xFF) << 24) |
                    (nibble_swap((temp >> 8) & 0xFF) << 16) |
                    (nibble_swap(temp & 0xFF) << 8) |
                    nibble_swap((temp >> 24) & 0xFF))
            temp ^= Rcon[i // Nk] << 24
        key_schedule.append(key_schedule[i - Nk] ^ temp)
    return key_schedule

def bytes_to_state(block):
    """Convert a 16-byte block into a 4x4 state matrix, mapping bytes column-wise."""
    state = [[0]*4 for _ in range(4)]
    for i in range(16):
        state[i % 4][i // 4] = block[i]
    return state

def state_to_bytes(state):
    """Convert a 4x4 state matrix into a 16-byte block, mapping bytes column-wise."""
    block = []
    for i in range(4):
        for j in range(4):
            block.append(state[j][i])
    return bytes(block)

def encrypt_block(block, key_schedule, timings):
    """Encrypt a single 16-byte block."""
    state = bytes_to_state(block)
    print(f"State dimensions: {len(state)}x{len(state[0])}")
    print("State before initial_add_round_key:", state)
    
    timings['initial_add_round_key'] -= time.perf_counter()
    state = add_round_key(state, key_schedule[0:4])
    timings['initial_add_round_key'] += time.perf_counter()

    for round in range(1, Nr):
        timings['sub_bytes'] -= time.perf_counter()
        state = sub_bytes(state)
        timings['sub_bytes'] += time.perf_counter()

        timings['shift_rows'] -= time.perf_counter()
        state = shift_rows(state)
        timings['shift_rows'] += time.perf_counter()

        timings['mix_columns'] -= time.perf_counter()
        state = mix_columns(state)
        timings['mix_columns'] += time.perf_counter()

        timings['add_round_key'] -= time.perf_counter()
        state = add_round_key(state, key_schedule[round*4:(round+1)*4])
        timings['add_round_key'] += time.perf_counter()

    # Final round (without MixColumns)
    timings['sub_bytes'] -= time.perf_counter()
    state = sub_bytes(state)
    timings['sub_bytes'] += time.perf_counter()

    timings['shift_rows'] -= time.perf_counter()
    state = shift_rows(state)
    timings['shift_rows'] += time.perf_counter()

    timings['add_round_key'] -= time.perf_counter()
    state = add_round_key(state, key_schedule[Nr*4:(Nr+1)*4])
    timings['add_round_key'] += time.perf_counter()

    return state_to_bytes(state)

def decrypt_block(block, key_schedule, timings):
    """Decrypt a single 16-byte block."""
    state = bytes_to_state(block)
    timings['initial_add_round_key'] -= time.perf_counter()
    state = add_round_key(state, key_schedule[Nr*4:(Nr+1)*4])
    timings['initial_add_round_key'] += time.perf_counter()

    for round in range(Nr - 1, 0, -1):
        timings['inv_shift_rows'] -= time.perf_counter()
        state = inv_shift_rows(state)
        timings['inv_shift_rows'] += time.perf_counter()

        timings['inv_sub_bytes'] -= time.perf_counter()
        state = inv_sub_bytes(state)
        timings['inv_sub_bytes'] += time.perf_counter()

        timings['add_round_key'] -= time.perf_counter()
        state = add_round_key(state, key_schedule[round*4:(round+1)*4])
        timings['add_round_key'] += time.perf_counter()

        timings['inv_mix_columns'] -= time.perf_counter()
        state = inv_mix_columns(state)
        timings['inv_mix_columns'] += time.perf_counter()

    # Initial round (without InvMixColumns)
    timings['inv_shift_rows'] -= time.perf_counter()
    state = inv_shift_rows(state)
    timings['inv_shift_rows'] += time.perf_counter()

    timings['inv_sub_bytes'] -= time.perf_counter()
    state = inv_sub_bytes(state)
    timings['inv_sub_bytes'] += time.perf_counter()

    timings['add_round_key'] -= time.perf_counter()
    state = add_round_key(state, key_schedule[0:4])
    timings['add_round_key'] += time.perf_counter()

    return state_to_bytes(state)

def pad(data):
    """Apply PKCS#7 padding."""
    padding_len = 16 - (len(data) % 16)
    return data + bytes([padding_len] * padding_len)

def unpad(data):
    """Remove PKCS#7 padding."""
    padding_len = data[-1]
    return data[:-padding_len]

def encrypt_file(input_file, output_file, key):
    """Encrypt the contents of input_file and write to output_file."""
    with open(input_file, 'rb') as f:
        data = f.read()

    data = pad(data)
    print(f"Data length after padding: {len(data)} bytes")
    key_schedule_timings = {'key_expansion': 0}
    timings = {
        'initial_add_round_key': 0,
        'sub_bytes': 0,
        'shift_rows': 0,
        'mix_columns': 0,
        'add_round_key': 0,
    }

    key_schedule_timings['key_expansion'] -= time.perf_counter()
    key_schedule = key_expansion(key)
    key_schedule_timings['key_expansion'] += time.perf_counter()

    encrypted_data = b''
    total_time = -time.perf_counter()
    for i in range(0, len(data), 16):
        block = data[i:i+16]
        print(f"Processing block of size: {len(block)} bytes")
        encrypted_block = encrypt_block(block, key_schedule, timings)
        encrypted_data += encrypted_block
    total_time += time.perf_counter()

    with open(output_file, 'wb') as f:
        f.write(encrypted_data)

    return total_time, timings, key_schedule_timings

def decrypt_file(input_file, output_file, key):
    """Decrypt the contents of input_file and write to output_file."""
    with open(input_file, 'rb') as f:
        data = f.read()

    key_schedule_timings = {'key_expansion': 0}
    timings = {
        'initial_add_round_key': 0,
        'inv_sub_bytes': 0,
        'inv_shift_rows': 0,
        'inv_mix_columns': 0,
        'add_round_key': 0,
    }

    key_schedule_timings['key_expansion'] -= time.perf_counter()
    key_schedule = key_expansion(key)
    key_schedule_timings['key_expansion'] += time.perf_counter()

    decrypted_data = b''
    total_time = -time.perf_counter()
    for i in range(0, len(data), 16):
        block = data[i:i+16]
        decrypted_block = decrypt_block(block, key_schedule, timings)
        decrypted_data += decrypted_block
    total_time += time.perf_counter()

    decrypted_data = unpad(decrypted_data)

    with open(output_file, 'wb') as f:
        f.write(decrypted_data)

    return total_time, timings, key_schedule_timings

def main():
    key = b'This is a key123'  # 16-byte key for AES-128

    # List of file sizes to test (in bytes)
    file_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB


    for size in file_sizes:
        # Create a test file of the specified size
        test_filename = f'test_{size}.bin'
        with open(test_filename, 'wb') as f:
            f.write(os.urandom(size))

        print(f'\nEncrypting and decrypting file of size: {size} bytes')

        # Encrypt the file
        enc_output_file = f'encrypted_{size}.bin'
        enc_total_time, enc_timings, enc_key_schedule_timings = encrypt_file(test_filename, enc_output_file, key)
        print(f'Encryption total time: {enc_total_time:.6f} seconds')

        # Decrypt the file
        dec_output_file = f'decrypted_{size}.bin'
        dec_total_time, dec_timings, dec_key_schedule_timings = decrypt_file(enc_output_file, dec_output_file, key)
        print(f'Decryption total time: {dec_total_time:.6f} seconds')

        # Verify correctness
        with open(test_filename, 'rb') as f1, open(dec_output_file, 'rb') as f2:
            if f1.read() == f2.read():
                print('Decryption successful, original and decrypted files match.')
            else:
                print('Decryption failed, files do not match.')

        # Display phase timings
        print('\nEncryption phase timings:')
        for phase, time_taken in enc_timings.items():
            print(f'  {phase}: {time_taken:.6f} seconds')
        print(f'  Key Expansion: {enc_key_schedule_timings["key_expansion"]:.6f} seconds')

        print('\nDecryption phase timings:')
        for phase, time_taken in dec_timings.items():
            print(f'  {phase}: {time_taken:.6f} seconds')
        print(f'  Key Expansion: {dec_key_schedule_timings["key_expansion"]:.6f} seconds')

        # Clean up test files (optional)
        os.remove(test_filename)
        os.remove(enc_output_file)
        os.remove(dec_output_file)

if __name__ == '__main__':
    main()
