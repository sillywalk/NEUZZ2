import os
import sys
import glob
import pickle

class DataHandler:
    def __init__(self, )

# get vector representation of input
def process_data():
    global MAX_BITMAP_SIZE
    global MAX_FILE_SIZE
    global SPLIT_RATIO
    global seed_list
    global new_seeds

    # shuffle training samples
    seed_list = glob.glob('./seeds/*')
    seed_list.sort()
    SPLIT_RATIO = len(seed_list)
    rand_index = np.arange(SPLIT_RATIO)
    np.random.shuffle(seed_list)
    new_seeds = glob.glob('./seeds/id_*')

    call = subprocess.check_output

    # get MAX_FILE_SIZE
    cwd = os.getcwd()
    max_file_name = call(['ls', '-S', cwd + '/seeds/']
                         ).split('\n')[0].rstrip('\n')
    MAX_FILE_SIZE = os.path.getsize(cwd + '/seeds/' + max_file_name)

    # create directories to save label, spliced seeds, variant length seeds, crashes and mutated seeds.
    if os.path.isdir("./bitmaps/") == False:
        os.makedirs('./bitmaps')
    if os.path.isdir("./splice_seeds/") == False:
        os.makedirs('./splice_seeds')
    if os.path.isdir("./vari_seeds/") == False:
        os.makedirs('./vari_seeds')
    if os.path.isdir("./crashes/") == False:
        os.makedirs('./crashes')

    # obtain raw bitmaps
    raw_bitmap = {}
    tmp_cnt = []
    out = ''
    for f in seed_list:
        tmp_list = []
        try:
            # append "-o tmp_file" to strip's arguments to avoid tampering tested binary.
            if argvv[0] == './strip':
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout',
                            '-m', '512', '-t', '500'] + argvv + [f] + ['-o', 'tmp_file'])
            else:
                out = call(['./afl-showmap', '-q', '-e', '-o',
                            '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f])
        except subprocess.CalledProcessError:
            print("find a crash")
        for line in out.splitlines():
            edge = line.split(':')[0]
            tmp_cnt.append(edge)
            tmp_list.append(edge)
        raw_bitmap[f] = tmp_list
    counter = Counter(tmp_cnt).most_common()

    # save bitmaps to individual numpy label
    label = [int(f[0]) for f in counter]
    bitmap = np.zeros((len(seed_list), len(label)))
    for idx, i in enumerate(seed_list):
        tmp = raw_bitmap[i]
        for j in tmp:
            if int(j) in label:
                bitmap[idx][label.index((int(j)))] = 1

    # label dimension reduction
    fit_bitmap = np.unique(bitmap, axis=1)
    print("data dimension" + str(fit_bitmap.shape))

    # save training data
    MAX_BITMAP_SIZE = fit_bitmap.shape[1]
    for idx, i in enumerate(seed_list):
        file_name = "./bitmaps/" + i.split('/')[-1]
        np.save(file_name, fit_bitmap[idx])

# training data generator


def generate_training_data(lb, ub):
    seed = np.zeros((ub - lb, MAX_FILE_SIZE))
    bitmap = np.zeros((ub - lb, MAX_BITMAP_SIZE))
    for i in range(lb, ub):
        tmp = open(seed_list[i], 'r').read()
        ln = len(tmp)
        if ln < MAX_FILE_SIZE:
            tmp = tmp + (MAX_FILE_SIZE - ln) * '\0'
        seed[i - lb] = [ord(j) for j in list(tmp)]

    for i in range(lb, ub):
        file_name = "./bitmaps/" + seed_list[i].split('/')[-1] + ".npy"
        bitmap[i - lb] = np.load(file_name)
    return seed, bitmap

def vectorize_file(fl):
    seed = np.zeros((1, MAX_FILE_SIZE))
    tmp = open(fl, 'r').read()
    ln = len(tmp)
    if ln < MAX_FILE_SIZE:
        tmp = tmp + (MAX_FILE_SIZE - ln) * '\0'
    seed[0] = [ord(j) for j in list(tmp)]
    seed = seed.astype('float32') / 255
    return seed

# splice two seeds to a new seed


def splice_seed(fl1, fl2, idxx):
    tmp1 = open(fl1, 'r').read()
    ret = 1
    randd = fl2
    while(ret == 1):
        tmp2 = open(randd, 'r').read()
        if len(tmp1) >= len(tmp2):
            lenn = len(tmp2)
            head = tmp2
            tail = tmp1
        else:
            lenn = len(tmp1)
            head = tmp1
            tail = tmp2
        f_diff = 0
        l_diff = 0
        for i in range(lenn):
            if tmp1[i] != tmp2[i]:
                f_diff = i
                break
        for i in reversed(range(lenn)):
            if tmp1[i] != tmp2[i]:
                l_diff = i
                break
        if f_diff >= 0 and l_diff > 0 and (l_diff - f_diff) >= 2:
            splice_at = f_diff + random.randint(1, l_diff - f_diff - 1)
            head = list(head)
            tail = list(tail)
            tail[:splice_at] = head[:splice_at]
            with open('./splice_seeds/tmp_' + str(idxx), 'w') as f:
                f.write("".join(tail))
            ret = 0
        print((f_diff, l_diff))
        randd = random.choice(seed_list)
