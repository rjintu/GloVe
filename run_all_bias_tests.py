from os import system

regions = ['midwest', 'south', 'northeast', 'west', 'federal']
times = ['1860-1889', '1890-1919', '1920-1949', '1950-1979', '1980-now']

for region in regions:
    for time in times:
        try:
            command = f"python bias_test.py --vocab_file vocab_{region}_opinions_{time}_formatted.txt --vectors_file vectors_{region}_opinions_{time}_formatted.txt --randomized_trials 1000 > output_{region}_{time}.txt"
            system(command)

        except FileNotFoundError:
            print('skipping...')
            continue
