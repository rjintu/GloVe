# create randomized corpora

# idea: create a dictionary/list of all of the potential sources of data
# create a maximum size, and randomly select within a range for what that size should be
# create a random order of data sources to select from
# within each data source, randomly seek to a certain position 
# then, pull a random amount from that part, perhaps some maximum so that we take data from a certain number of places
# append into the corresponding file, repeat until the file has sufficient data

import os, random, gc, argparse

filenames = []
in_dir = r'/Users/Firebolt/Documents/Sem5-Fall2021/cos397/GloVe/cleaned_data/cleaned_completed'
out_dir = r'/Users/Firebolt/Documents/Sem5-Fall2021/cos397/GloVe/cleaned_data/randomized'

def generate_corpus(output_file):
    for filename in os.listdir(in_dir):
        if filename.endswith('.txt'):
            filenames.append(filename)

    # randomly order the list
    random.shuffle(filenames)

    output = open(os.path.join(out_dir, output_file), 'w')
    output_lines = []

    # MAXIMUM_SIZE = random.randint(1000, 10000) # TODO: remove this, just for testing
    MAXIMUM_SIZE = random.randint(1073741824, 2147483648) # number of bytes in 1 GB or 2 GB
    bytes_written = 0


    for filename in filenames:
        if bytes_written >= MAXIMUM_SIZE:
            break
        source = open(os.path.join(in_dir, filename), 'r')

        # randomly seek to a certain position
        seek_lines = random.randint(0, 5000000)
        num_lines_read = 0

        line = source.readline()

        # seek to the right place in the file 
        while line and num_lines_read < seek_lines:
            num_lines_read += 1
            line = source.readline()

        # copy random number of lines
        num_lines_copy = random.randint(0, 1000000)
        num_lines_read = 0
        # start copying lines, ensure we don't exceed bytes
        while line and num_lines_read < num_lines_copy and bytes_written < MAXIMUM_SIZE:
            num_lines_read += 1
            output_lines.append(line)
            bytes_written += len(line)

            if len(output_lines) >= 20000:
                output.writelines(output_lines)
                del output_lines
                output_lines = []
                gc.collect()
                print(num_lines_read)

            line = source.readline()
        
        # copy any extraneous lines
        if len(output_lines) > 0:
            output.writelines(output_lines)
            del output_lines
            output_lines = []
            gc.collect()

        source.close()

    output.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_files', nargs=1, metavar=('NUM_FILES'))
    parser.add_argument('--seed', nargs=1)
    args = parser.parse_args()

    if args.seed:
            random.seed(int(args.seed))
    else:
        random.seed()

    for i in range(int(args.num_files[0])):
        generate_corpus(f'output_{i+15}.txt') # guarantee no file overwriting

if __name__ == '__main__':
    main()