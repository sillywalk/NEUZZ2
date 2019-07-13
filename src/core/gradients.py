"""The main code for performing gradient guided mutation.

Gradient contains methods to format and the input data, create auxillary
directories to save some information used for multiple rounds, etc. It also
allows creates training data for the Deep neural smoothing model, etc. The key
hook to external code is the gen_grad method that ties everything together.

Adapted (almost verbatim in several places) from Dongdong's original code.
For code, see: https://github.com/dongdong/neuzz
For paper, see: https://arxiv.org/abs/1807.05620
"""

import os
import sys
import math
import keras
import shlex
import pickle
import random
import numpy as np
from time import time
from copy import copy
from glob import glob
import keras.backend as K
from pathlib import Path, PosixPath
from typing import Tuple, List, NewType
from collections import Counter, defaultdict
# Update path information
cur = Path.cwd()
while cur.name != "src":
cur = cur.parent
if not cur in sys.path:
    sys.path.append(str(cur))

from utils.parse_config import Config
from core.neural_network import NeuralNetwork

__author__ = "Rahul Krishna and Dongdong She"
__copyright__ = "Copyright 2019"
__credits__ = ["Dongdong She"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Rahul Krishna"
__email__ = "i.m.ralk@gmail.com"
__status__ = "Research Prototype"

# Hinting support for user defined types
Numpy2D = NewType("Numpy2D", np.ndarray)
KerasModel = NewType("KerasModel", keras.engine.sequential.Sequential)


class Gradient:
    def __init__(self, config: Config, argv: list, verbose: bool = False) -> None:
        """
        The primary container with all the components to perform a gradient guided mutation.

        Parameters
        ----------
        config: Config
            A shared configurations container.
        argv: str
            System arguments
        verbose: Bool
            Turns on/off verbose mode.
        """
        # Initialize variables
        self.timer: int = 0
        self.argv: list = argv
        self.round_count: int = 0
        self.verbose: bool = verbose
        self.usr_config: Config = copy(config)
        # Add some target specific config.
        self._init_additional_config()

    def _init_additional_config(self) -> None:
        """
        Initialize additional parameters.
        """
        self.aux_config = Config()
        self.aux_config.set_config({
            "core": {
                "MAX_FILE_SIZE": 0,
                "MAX_BITMAP_SIZE": 0,
                "NUM_SEEDS": 0
            }
        })

    @staticmethod
    def _os_cmd_call(cmd: str) -> str:
        """
        Run a command on the commandline

        Parameters
        ----------
        cmd: str
            Command as a string

        Returns
        -------
        str:
            What ever is retured when the command is run
        """
        return subprocess.check_output(shlex.split(cmd_str))

    self._create_intermediate_directories(self) -> None:
        """
        Create directories to save intermediate files. These include:
         + bitmaps,
         + crashes,
         + variant length seeds, and
         + Gradient information of the seeds.
        """

        bitmap_path = self.cwd.joinpath("bitmaps")
        if not bitmap_path.isdir():
            bitmap_path.mkdir()

        crashes_path = self.cwd.joinpath("crashes")
        if not crashes_path.isdir():
            crashes_path.mkdir()

        vari_seed_path = self.cwd.joinpath("vari_seeds")
        if not vari_seed_path.isdir():
            vari_seed_path.mkdir()

        gradient_info_path = self.cwd.joinpath("gradient_info")
        if not gradient_info_path.isdir():
            gradient_info_path.mkdir()

    def _process_data(self) -> None:
        """
        Process the seeds to generate a vector representation of the input.
        """
        # Obtain a file form the seeds directory
        self.cwd: pathlib.PosixPath = Path.cwd()

        # Gather the seeds
        self.seed_list: Path.glob = self.cwd.joinpath('seeds').glob("*")

        # Sort the seeds from largest to smallest by file size
        self.seed_list: List = sorted(
            self.seed_list, key=lambda x: os.path.getsize(x), reverse=True)

        # Update new seed list
        # Note: New seeds will be stored with a prefix of round number
        self.new_seeds: Path.glob = self.cwd.joinpath(
            'seeds').glob('{}_*'.format(self.round_count))

        # Sort the seeds from largest to smallest by file size
        self.new_seeds: List = sorted(
            self.new_seeds, key=lambda x: os.path.getsize(x), reverse=True)

        # Update the seed length
        self.aux_config.core.NUM_SEEDS = len(self.seed_list)

        # TODO: What's this for?
        rand_index = np.arange(self.aux_config.core.NUM_SEEDS)

        # Find the size of the largest seed file and set MAX_FILE_SIZE config
        largest_seed_file = self.seed_list[0]
        self.aux_config.core.MAX_FILE_SIZE = os.path.getsize(largest_seed_file)

        # Shuffle the seeds
        np.random.shuffle(self.seed_list)

        self._create_intermediate_directories()

        # ----------------------------------------------------------------------
        # Run AFL to collect the edge coverage data.
        # This is a major resource hog :(
        # ----------------------------------------------------------------------
        # To store the bitmaps of the seeds
        seed_bitmap_dict: defaultdict = defaultdict(list)
        edge_list_all: list = []  # Edge IDs of all the seeds
        out: str = ''
        for seed_file in self.seed_list:
            try:
                # append "-o tmp_file" to strip's arguments to avoid tampering tested binary.
                arg_str = " ".join(self.argv)
                if self.argv[0] == './strip':
                    out = self._os_cmd_call(
                        './afl-showmap -q -e -o /dev/stdout -m 512 -t 500 {} {} -o tmp_file'.format(arg_str, str(seed_file)).split(' '))
                else:
                    out = self._os_cmd_call(
                        './afl-showmap -q -e -o /dev/stdout -m 512 -t 500 {} {}'.format(arg_str, str(seed_file)).split(' '))

            except subprocess.CalledProcessError:
                if self.verbose:
                    print("Crash Detected")
                else:
                    pass

            for line in out.splitlines():
                # "out" looks like EdgeID:EdgeCount
                edge_id: str = line.split(':')[0]
                # Add current edge ID to the overall collection
                edge_list_all.append(edge_id)
                # Add current edge ID to the current seed's collection
                seed_bitmap_dict[seed_file.name].append(edge_id)

        # Count and sort in decreasing order the number of times an edge's taken
        edge_id_counts: List(Tuple(str, int)) = Counter(
            edge_list_all).most_common()
        # save bitmaps to individual numpy label
        label: List[int] = [int(edge_id) for edge_id, __ in edge_id_counts]
        # A 2-D numpy array
        bitmap: Numpy2D = np.zeros((len(self.seed_list), len(label)))

        # ----------------------------------------------------------------------
        # Populate a 2D Bitmap
        # ````````````````````
        # E.g.,
        #      Let's say we have
        #         (a) 3 seed files, and
        #         (b) 5 possible edges.
        #     Initially, we have
        #         label = [0,1,2,3,4]
        #         Bitmap = [
        #             [0, 0, 0, 0],
        #             [0, 0, 0, 0],
        #             [0, 0, 0, 0]
        #         ]
        #     If
        #         - Seed 1 activates edges 0, 1, 4
        #         - Seed 2 activates edges 1, 3, 4
        #         - Seed 3 activates edges 2, 3,
        # ----------------------------------------------------------------------
        for idx, seed_file in enumerate(self.seed_list):
            seed_edges = seed_bitmap_dict[seed_file.name]
            for edge in seed_edges:
                if int(edge) in label:
                    bitmap[idx][label.index((int(edge)))] = 1
        # ----------------------------------------------------------------------
        # Merge columns (edge_ids) that look exactly the same.
        # ````````````````````````````````````````````````````
        # Recall what our bitmap looked like?
        #     Bitmap = [
        #         [1, 1, 0, 0, 1],
        #         [0, 1, 0, 1, 1],
        #         [0, 0, 1, 1, 0]
        #     ]
        # See how columns 1 and 4 are exactly alike? We should merge them.
        # ----------------------------------------------------------------------
        fit_bitmap: Numpy2D = np.unique(bitmap, axis=1)
        if self.verbose:
            print("data dimension" + str(fit_bitmap.shape))

        # Update the maximum bitmap size
        self.aux_config.core.MAX_BITMAP_SIZE = fit_bitmap.shape[1]

        # Save the bitmap data.
        # This will be used later on to train the neural network
        for idx, seed_file in enumerate(self.seed_list):
            file_name = self.cwd.joinpath("/bitmaps/", seed_file.name)
            np.save(file_name, fit_bitmap[idx])

    def _compute_gradients(self, edge_id, chosen_seeds, use_sign):
        adv_list = []

        layer_list = [(layer.name, layer)
                      for layer in self.smoothed_model.layers]

        # Compute the loss of all the outputs with
        # respect to the edge "edge_id"
        # TODO: layer_list[-2] --> layer_list[0]
        loss = layer_list[-2][1].output[:, edge_id]
        grads = K.gradients(loss, self.model.input)[0]
        iterate = K.function([self.model.input], [loss, grads])
        ll = 2
        while(chosen_seeds[0] == chosen_seeds[1]):
            chosen_seeds[1] = random.choice(self.seed_list)

        for index in range(ll):
            x = vectorize_file(chosen_seeds[index])
            loss_value, grads_value = iterate([x])
            idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[
                :, -MAX_FILE_SIZE:].reshape((MAX_FILE_SIZE,)), 0)
            if use_sign:
                val = np.sign(grads_value[0][idx])
            else:
                val = np.random.choice([1, -1], MAX_FILE_SIZE, replace=True)
            adv_list.append((idx, val, chosen_seeds[index]))

        return adv_list

    def _gen_mutate(self, edge_num: int, sign: bool = True) -> None:
        """
        Grenerate information to guide muatation

        Parameters
        ----------
        model: KerasModel
            Trained neural network model
        edge_num: int
            Length of the output bitmaps
        sign: bool (optional)
            Consider the sign of the gradients. Default: True.
        """
        if self.verbose:
            print(":: Debug :::: Round #{}".format(self.round_count))

        tmp_list = []

        # Select seeds to be used for this round.
        if(self.round_count == 0):
            new_seed_list = self.seed_list
        else:
            new_seed_list = self.new_seeds

        if len(new_seed_list) < edge_num:
            shuffled_seeds_1 = [new_seed_list[i] for i in np.random.choice(
                len(new_seed_list), edge_num, replace=True)]
            shuffled_seeds_2 = [new_seed_list[i] for i in np.random.choice(
                len(new_seed_list), edge_num, replace=True)]
        else:
            shuffled_seeds = [new_seed_list[i] for i in np.random.choice(
                len(new_seed_list), edge_num, replace=False)]
            shuffled_seeds_2 = [new_seed_list[i] for i in np.random.choice(
                len(new_seed_list), edge_num, replace=True)]

        # select output neurons to compute gradient
        interested_indice = np.random.choice(
            self.aux_config.core.MAX_BITMAP_SIZE, edge_num)

        # TODO: Save the gradient files separately.
        with open('gradient_info_p', 'w') as f:
            for idxx, edge_id in enumerate(interested_indice):
                # Keras stalls after multiple gradient compuations.
                # We release the memory and rebuild the model.
                if (idxx % 100 == 0):
                    del self.smoothed_model
                    self.smoothed_model = self.neural_net.clear_and_rebuild_model(
                        weights="hard_label.h5")

                if self.verbose:
                    print("Edge number" + str(idxx))

                chosen_seeds = [
                    shuffled_seeds_1[idxx],
                    shuffled_seeds_2[idxx]
                ]

                adv_list = self._compute_gradients(
                    edge_id,
                    chosen_seeds,
                    use_sign=sign)

                tmp_list.append(adv_list)

                for ele in adv_list:
                    ele0 = [str(el) for el in ele[0]]
                    ele1 = [str(int(el)) for el in ele[1]]
                    ele2 = ele[2]
                    f.write(",".join(ele0) + '|' +
                            ",".join(ele1) + '|' + ele2 + "\n")

    def generate_gradients(self, data: str) -> self:
        """
        The main interface of the gradient computation. Handles everything from preprocessing data, to building a NN model, to generating mutants.

        Parameters
        ----------
        data: str
            Operation mode. "train" -> use sign; "sloww" -> no sign.
        """
        self.timer = time()
        self._process_data()

        self.neural_net = NeuralNetwork(
            seed_list=self.seed_list,
            bitmaps_dir=self.bitmap_path,
            usr_config=self.usr_config.model,
            aux_config=self.aux_config)

        self.smoothed_model = self.neural_net.build_and_train_smoothing_model()

        # Determine bitmap length
        bitmap_len = self.usr_config.fuzz.bitmap_len

        # If the user defined bitmap length is zero, then
        # we want to use the max_bitmap_len
        if bitmap_len == 0:
            bitmap_len = self.aux_config.core.MAX_BITMAP_SIZE

        if(data[:5] == "train"):
            self._gen_mutate(num_edges=bitmap_len)
        else:
            self._gen_mutate(num_edges=bitmap_len, sign=False)

        self.round_count += 1
        self.timer = time() - self.timer
        if self.verbose:
            print(self.timer)

        return self
