import time               # Importa o módulo time para medições de tempo
from progressbar import progressbar  # Importa a função progressbar para exibição de barra de progresso
import cython             # Importa o cython (usado para otimizações de código)
import sys                # Importa o sys para ler argumentos de linha de comando

# Parâmetros AES
Nb = 4  # Número de colunas que compõem o Estado (State). Para AES, Nb = 4
Nk = 4  # Número de palavras (32 bits) que compõem a chave. Para AES-128, Nk = 4
Nr = 10 # Número de rodadas. Para AES-128, Nr = 10

# Constantes AES
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

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def nibble_swap(byte):
    """Troca o nibble alto com o nibble baixo de um byte."""
    # Extrai o nibble alto (4 bits superiores)
    high_nibble = (byte & 0xF0) >> 4
    # Extrai o nibble baixo (4 bits inferiores) e o desloca para a parte alta
    low_nibble = (byte & 0x0F) << 4
    # Combina os nibbles invertidos
    return high_nibble | low_nibble

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sub_bytes(state):
    """Aplica a substituição nibble_swap em cada byte do estado."""
    # Percorre cada elemento da matriz estado (4x4)
    for i in range(4):
        for j in range(4):
            # Aplica nibble_swap em cada byte do estado
            state[i][j] = nibble_swap(state[i][j])
    return state

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def inv_sub_bytes(state):
    """A operação inversa do nibble_swap é a mesma que o nibble_swap."""
    # Como o nibble_swap é involutivo (aplicando duas vezes volta ao original),
    # a inversa é a própria função sub_bytes.
    return sub_bytes(state)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def shift_rows(state):
    """Desloca as linhas do estado para a esquerda."""
    # A linha 0 não é deslocada
    # Desloca a linha 1 para a esquerda em 1 byte
    state[1] = state[1][1:] + state[1][:1]  
    # Desloca a linha 2 para a esquerda em 2 bytes
    state[2] = state[2][2:] + state[2][:2]
    # Desloca a linha 3 para a esquerda em 3 bytes
    state[3] = state[3][3:] + state[3][:3]
    return state

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def inv_shift_rows(state):
    """Desloca as linhas do estado para a direita (operação inversa)."""
    # A linha 0 não é deslocada
    # Desloca a linha 1 para a direita em 1 byte
    state[1] = state[1][-1:] + state[1][:-1]
    # Desloca a linha 2 para a direita em 2 bytes
    state[2] = state[2][-2:] + state[2][:-2]
    # Desloca a linha 3 para a direita em 3 bytes
    state[3] = state[3][-3:] + state[3][:-3]
    return state

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xtime(a):
    """Multiplica por x (ou seja, {02}) em GF(2^8)."""
    # Se o bit mais significativo estiver setado, faz XOR com 0x1B após deslocar
    # caso contrário, apenas desloca
    return (((a << 1) & 0xFF) ^ 0x1B) if (a & 0x80) else (a << 1)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mix_single_column(a):
    """Realiza a operação MixColumns em uma única coluna."""
    # Calcula t como XOR de todos os bytes da coluna
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    # Guarda o valor original do primeiro elemento da coluna
    u = a[0]
    # Aplica a transformação MixColumns em cada elemento
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)
    return a

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mix_columns(state):
    """Aplica a operação MixColumns em todas as colunas do estado."""
    for i in range(4):
        # Extrai a coluna i
        col = [state[j][i] for j in range(4)]
        # Mistura a coluna
        col = mix_single_column(col)
        # Devolve a coluna misturada ao estado
        for j in range(4):
            state[j][i] = col[j]
    return state

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def multiply(a, b):
    """Multiplica dois valores no campo GF(2^8)."""
    p = 0
    # Executa 8 vezes (para cada bit)
    for _ in range(8):
        # Se o bit menos significativo de b estiver setado, faz XOR com a
        p ^= (b & 1) * a
        # Desloca a para a esquerda, se o bit mais significativo estava setado, faz XOR com 0x1B
        a = ((a << 1) ^ (0x1B if a & 0x80 else 0)) & 0xFF
        # Desloca b para a direita
        b >>= 1
    return p

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def generate_lookup_table(constant):
    """Gera uma tabela de busca para multiplicação no campo GF(2^8) com uma constante fixa."""
    table = []
    # Multiplica todos os valores de 0 a 255 pela constante e armazena
    for x in range(256):
        table.append(multiply(x, constant))
    return table

# Gera tabelas de busca para as constantes utilizadas em inv_mix_columns
mul_9 = generate_lookup_table(9)
mul_11 = generate_lookup_table(11)
mul_13 = generate_lookup_table(13)
mul_14 = generate_lookup_table(14)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def inv_mix_columns(state):
    """Passo Inverse MixColumns usando tabelas pré-computadas."""
    for i in range(4):
        # Extrai a coluna i
        s = [state[j][i] for j in range(4)]
        # Aplica a transformação inversa usando as tabelas pré-computadas
        state[0][i] = mul_14[s[0]] ^ mul_11[s[1]] ^ mul_13[s[2]] ^ mul_9[s[3]]
        state[1][i] = mul_9[s[0]] ^ mul_14[s[1]] ^ mul_11[s[2]] ^ mul_13[s[3]]
        state[2][i] = mul_13[s[0]] ^ mul_9[s[1]] ^ mul_14[s[2]] ^ mul_11[s[3]]
        state[3][i] = mul_11[s[0]] ^ mul_13[s[1]] ^ mul_9[s[2]] ^ mul_14[s[3]]
    return state

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def add_round_key(state, round_key_words):
    """Adiciona (XOR) a round key ao estado."""
    for c in range(4):
        # Extrai os bytes da palavra da chave para a coluna c
        rk_bytes = [
            (round_key_words[c] >> 24) & 0xFF,
            (round_key_words[c] >> 16) & 0xFF,
            (round_key_words[c] >> 8) & 0xFF,
            round_key_words[c] & 0xFF,
        ]
        # Aplica XOR de cada byte da chave com o estado
        for r in range(4):
            state[r][c] ^= rk_bytes[r]
    return state

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def key_expansion(key):
    """Gera a chave expandida a partir da chave principal."""
    # Converte a chave em uma lista de inteiros
    key_symbols = [k for k in key]
    # Preenche com 0x01 se a chave tiver menos de 16 bytes
    if len(key_symbols) < 4 * Nk:
        key_symbols += [0x01] * (4 * Nk - len(key_symbols))
    key_schedule = []
    # Cria as palavras iniciais a partir da chave
    for i in range(Nk):
        word = (key_symbols[4*i] << 24) | (key_symbols[4*i+1] << 16) | (key_symbols[4*i+2] << 8) | key_symbols[4*i+3]
        key_schedule.append(word)
    # Expande as palavras para gerar todas as chaves de rodada
    for i in range(Nk, Nb * (Nr + 1)):
        temp = key_schedule[i - 1]
        if i % Nk == 0:
            # RotWord + SubWord (usando nibble_swap) + XOR com Rcon
            temp = ((nibble_swap((temp >> 16) & 0xFF) << 24) |
                    (nibble_swap((temp >> 8) & 0xFF) << 16) |
                    (nibble_swap(temp & 0xFF) << 8) |
                    nibble_swap((temp >> 24) & 0xFF))
            temp ^= Rcon[i // Nk] << 24
        key_schedule.append(key_schedule[i - Nk] ^ temp)
    return key_schedule

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def bytes_to_state(block):
    """Converte um bloco de 16 bytes em uma matriz estado 4x4, mapeando bytes coluna a coluna."""
    state = [[0]*4 for _ in range(4)]
    # Preenche o estado coluna a coluna
    for i in range(16):
        state[i % 4][i // 4] = block[i]
    return state

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def state_to_bytes(state):
    """Converte a matriz estado 4x4 em um bloco de 16 bytes, mapeando coluna a coluna."""
    block = []
    # Lê coluna a coluna e converte em um array de bytes
    for i in range(4):
        for j in range(4):
            block.append(state[j][i])
    return bytes(block)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def encrypt_block(block, key_schedule, timings):
    """Encripta um bloco de 16 bytes."""
    # Converte o bloco de bytes para o estado (matriz 4x4)
    state = bytes_to_state(block)
    
    # Etapa inicial: AddRoundKey com a chave da rodada 0
    timings['initial_add_round_key'] -= time.perf_counter()
    state = add_round_key(state, key_schedule[0:4])
    timings['initial_add_round_key'] += time.perf_counter()

    # Executa Nr-1 rodadas intermediárias
    for round in range(1, Nr):
        # SubBytes
        timings['sub_bytes'] -= time.perf_counter()
        state = sub_bytes(state)
        timings['sub_bytes'] += time.perf_counter()
        
        # ShiftRows
        timings['shift_rows'] -= time.perf_counter()
        state = shift_rows(state)
        timings['shift_rows'] += time.perf_counter()

        # MixColumns
        timings['mix_columns'] -= time.perf_counter()
        state = mix_columns(state)
        timings['mix_columns'] += time.perf_counter()

        # AddRoundKey
        timings['add_round_key'] -= time.perf_counter()
        state = add_round_key(state, key_schedule[round*4:(round+1)*4])
        timings['add_round_key'] += time.perf_counter()

    # Rodada final (sem MixColumns)
    timings['sub_bytes'] -= time.perf_counter()
    state = sub_bytes(state)
    timings['sub_bytes'] += time.perf_counter()

    timings['shift_rows'] -= time.perf_counter()
    state = shift_rows(state)
    timings['shift_rows'] += time.perf_counter()

    timings['add_round_key'] -= time.perf_counter()
    state = add_round_key(state, key_schedule[Nr*4:(Nr+1)*4])
    timings['add_round_key'] += time.perf_counter()

    # Converte o estado de volta para bytes
    return state_to_bytes(state)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def decrypt_block(block, key_schedule, timings):
    """Decripta um bloco de 16 bytes."""
    # Converte o bloco para estado
    state = bytes_to_state(block)

    # Primeira etapa: AddRoundKey com a chave da última rodada
    timings['initial_add_round_key'] -= time.perf_counter()
    state = add_round_key(state, key_schedule[Nr*4:(Nr+1)*4])
    timings['initial_add_round_key'] += time.perf_counter()

    # Executa Nr-1 rodadas intermediárias
    for round in range(Nr - 1, 0, -1):
        # InvShiftRows
        timings['inv_shift_rows'] -= time.perf_counter()
        state = inv_shift_rows(state)
        timings['inv_shift_rows'] += time.perf_counter()

        # InvSubBytes (mesmo que sub_bytes, pois nibble_swap é involutivo)
        timings['inv_sub_bytes'] -= time.perf_counter()
        state = sub_bytes(state)
        timings['inv_sub_bytes'] += time.perf_counter()

        # AddRoundKey
        timings['add_round_key'] -= time.perf_counter()
        state = add_round_key(state, key_schedule[round*4:(round+1)*4])
        timings['add_round_key'] += time.perf_counter()

        # InvMixColumns
        timings['inv_mix_columns'] -= time.perf_counter()
        state = inv_mix_columns(state)
        timings['inv_mix_columns'] += time.perf_counter()

    # Rodada inicial (sem InvMixColumns)
    timings['inv_shift_rows'] -= time.perf_counter()
    state = inv_shift_rows(state)
    timings['inv_shift_rows'] += time.perf_counter()

    timings['inv_sub_bytes'] -= time.perf_counter()
    state = sub_bytes(state)
    timings['inv_sub_bytes'] += time.perf_counter()

    timings['add_round_key'] -= time.perf_counter()
    state = add_round_key(state, key_schedule[0:4])
    timings['add_round_key'] += time.perf_counter()

    # Converte o estado de volta para bytes
    return state_to_bytes(state)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pad(data):
    """Aplica padding PKCS#7."""
    # Calcula quantos bytes faltam para completar um múltiplo de 16
    padding_len = 16 - (len(data) % 16)
    # Aplica padding (vários bytes com o valor da quantidade de padding)
    return data + bytes([padding_len] * padding_len)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def unpad(data):
    """Remove o padding PKCS#7."""
    # O último byte indica quantos bytes de padding foram adicionados
    padding_len = data[-1]
    # Remove o padding
    return data[:-padding_len]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def encrypt_file(input_file, output_file, key):
    """Encripta o conteúdo de input_file e escreve em output_file."""
    # Lê o arquivo de entrada em modo binário
    with open(input_file, 'rb') as f:
        data = f.read()

    # Aplica padding
    data = pad(data)

    # Dicionários para armazenar tempos de cada fase
    key_schedule_timings = {'key_expansion': 0}
    timings = {
        'initial_add_round_key': 0,
        'sub_bytes': 0,
        'shift_rows': 0,
        'mix_columns': 0,
        'add_round_key': 0,
    }

    # Mede o tempo de expansão da chave
    key_schedule_timings['key_expansion'] -= time.perf_counter()
    key_schedule = key_expansion(key)
    key_schedule_timings['key_expansion'] += time.perf_counter()

    # Mede o tempo total
    total_time = -time.perf_counter()
    # Abre o arquivo de saída em modo append binário e com buffering
    with open(output_file, 'ab', buffering=1024) as f:
        # Percorre cada bloco de 16 bytes
        for i in progressbar(range(0, len(data), 16)):
            block = data[i:i+16]
            # Encripta o bloco
            f.write(encrypt_block(block, key_schedule, timings))
        total_time += time.perf_counter()

    return total_time, timings, key_schedule_timings

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def decrypt_file(input_file, output_file, key):
    """Decripta o conteúdo de input_file e escreve em output_file."""
    # Lê o arquivo de entrada em binário
    with open(input_file, 'rb') as f:
        data = f.read()

    # Dicionários para armazenar tempos
    key_schedule_timings = {'key_expansion': 0}
    timings = {
        'initial_add_round_key': 0,
        'inv_sub_bytes': 0,
        'inv_shift_rows': 0,
        'inv_mix_columns': 0,
        'add_round_key': 0,
    }

    # Mede o tempo da expansão da chave
    key_schedule_timings['key_expansion'] -= time.perf_counter()
    key_schedule = key_expansion(key)
    key_schedule_timings['key_expansion'] += time.perf_counter()

    # Mede o tempo total
    total_time = -time.perf_counter()
    # Abre o arquivo de saída em modo append
    with open(output_file, 'ab') as f:
        # Percorre cada bloco de 16 bytes

        for i in progressbar(range(0, len(data), 16)):
            block = data[i:i+16]
            # Decripta o bloco
            f.write(decrypt_block(block, key_schedule, timings))
        total_time += time.perf_counter()
        
    # Remove o padding após a decriptação
    with open(output_file, 'rb') as f:
        decrypted_data = f.read()
        decrypted_data = unpad(decrypted_data)
        # Reescreve o arquivo sem o padding
        with open(output_file, 'wb') as f2:
            f2.write(decrypted_data)

    return total_time, timings, key_schedule_timings

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def main():
    # Define a chave de 16 bytes (AES-128)
    key = b'This is a key123'
    # Verifica se o usuário forneceu argumentos suficientes
    if(len(sys.argv) < 3):
        print(f'Entrada Inválida:\nEncriptar arquivo: python3 AES.py e <input_file> <encrypted_file>\nDecriptar arquivo: python3 AES.py d <input_file> <encrypted_file>')
        exit(1)
    # Obtém nomes de arquivos da linha de comando
    enc_output_file = sys.argv[3]
    dec_output_file = sys.argv[3]
    input_file_name = sys.argv[2]
    # Se o argumento for "e", encripta o arquivo
    if(sys.argv[1] == "e"):
        print(f'\nEncrypting file')
        enc_total_time, enc_timings, enc_key_schedule_timings = encrypt_file(input_file_name, enc_output_file, key)
        print(f'Encryption total time: {enc_total_time:.6f} seconds')
        print('\nEncryption phase timings:')
        for phase, time_taken in enc_timings.items():
            print(f'  {phase}: {time_taken:.6f} seconds')
        print(f'  Key Expansion: {enc_key_schedule_timings["key_expansion"]:.6f} seconds')

    # Se o argumento for "d", decripta o arquivo
    if(sys.argv[1] == "d"):
        print(f'\nDecrypting file')
        dec_total_time, dec_timings, dec_key_schedule_timings = decrypt_file(input_file_name, dec_output_file, key)
        print(f'Decryption total time: {dec_total_time:.6f} seconds')
        print('\nDecryption phase timings:')
        for phase, time_taken in dec_timings.items():
            print(f'  {phase}: {time_taken:.6f} seconds')
        print(f'  Key Expansion: {dec_key_schedule_timings["key_expansion"]:.6f} seconds')

if __name__ == '__main__':
    main()
