import math
from tensorflow.keras.utils import PyDataset
from pathlib import Path
from joblib import Parallel, delayed, parallel_config
import os

import torch
import numpy as np
from tqdm.notebook import tqdm

from evaluation.evaluator import read_ground_truth_files

from pathlib import Path
import itertools
from natsort import natsorted

from transformers import BertTokenizer, BertModel
from joblib import Memory

memory = Memory(".cache", verbose=0)
# memory = Memory(".cache_test", verbose=1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@memory.cache
def get_tokenizer_model():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    if DEVICE == "cuda":
        bert_model = BertModel.from_pretrained("bert-base-cased").to(DEVICE)
    else:
        bert_model = BertModel.from_pretrained("bert-base-cased")

    return bert_tokenizer, bert_model

@memory.cache
def read_problem_files(problem_folder, start=0, stop=None):
    # print(f"{start=} {stop=}")
    problems = []
    files = itertools.islice(natsorted(Path(problem_folder).glob('problem-*.txt')), start, stop)
    for problem_file in files:
        # print(f"{problem_file=}")
        # number = problem_file.name[len("problem-") : -len(".txt")]
        with open(problem_file, 'r', encoding="utf8") as fh:
            problems.append(fh.readlines())
    return problems

def pad_paragraph(paragraph_embedding, desired_length):
    d1, d2, d3 = paragraph_embedding.shape
    # print(f"{paragraph_embedding.shape=}")

    target = torch.zeros(d1, desired_length, d3)
    # print(f"{target.shape=}")
    target[:, :d2, :] = paragraph_embedding

    return target

def get_simple_ground_truth(ground_truth, problem_numbers):
    simple_ground_truth = []
    for num in problem_numbers:
        task_3_ground_truth = ground_truth[f"problem-{num}"]["paragraph-authors"]
        simple_ground_truth.append(task_3_ground_truth)
    return simple_ground_truth

def get_task_3_ground_truth(simple_ground_truth):
    # print(f"{len(simple_ground_truth)=}")
    task_gt = []
    for problem in simple_ground_truth:
        problem_gt = []
        for author1, author2 in itertools.combinations(problem, 2):
            problem_gt.append(int(author1 != author2))
        task_gt.append(problem_gt)
    # print(f"{len(task_gt)=}")
    return task_gt

# TODO: Invert the function get_task_3_ground_truth. Our model will output a bunch of binary labels which need to be converted to the task 3 ground truth format
# Ground truth format (gtf): [1, 2, 2, 2, 2, 3, 2, 2]
# Binary labels for comparisons (bl): [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
# Each binary label is the result of comparing two paragraphs. 1 means there was an author change, 0 means there was no author change
# For example, bl[0], is the result of comparing gtf[0]=1 and gtf[1]=2. 1 != 2, therefore bl[0] = 1. bl[1]=1 is the result of gtf[0] == gtf[2] (1 == 2)
def get_simple_ground_truth_from_task_3(task_3_ground_truth):
    simple_gt = []
    for problem in task_3_ground_truth:
        # k = n*(n-1)/2
        # n**2 - n - 2k = 0
        coeff = [1, -1, len(problem) * -2]
        roots = np.roots(coeff)
        gt_length = int(roots[roots > 0][0])
        # print(gt_length)

        gt = np.zeros(gt_length, dtype=np.uint8)
        gt[0] = 1
        for i in range(1, gt_length):
            # loop for gt[i]
            num_comparisons = i
            pointer = i - 1
            modified_flag = False
            # print(f"{i=} {num_comparisons=} {pointer=}")
            for gt_i, j in enumerate(range(gt_length-2, 1, -1)[:num_comparisons]):
                # comparison between gt[gt_i] and gt[i]
                # print(f"{gt_i=} {j=} {pointer=} {task_3_ground_truth[pointer]=}")
                bin_label = task_3_ground_truth[pointer]
                if bin_label == 0:
                    # print(f"{gt[i]=} {gt[gt_i]=}")
                    gt[i] = gt[gt_i]
                    modified_flag = True
                    break

                pointer += j
            if not modified_flag:
                # print(f"No modified")
                gt[i] = np.max(gt) + 1
            # print(f"{gt}\n")
        simple_gt.append(gt)
    return simple_gt

def get_cuda_memory(i):
    if not torch.cuda.is_available():
        return 0

    total_memory = torch.cuda.get_device_properties(i).total_memory
    reserved_memory = torch.cuda.memory_reserved(i)
    allocated_memory = torch.cuda.memory_allocated(i)
    free_memory = total_memory - (reserved_memory + allocated_memory)
    
    return free_memory / (1024 ** 2) # MB

def flatten_problems(problems_list, squeeze=False):
        # [print(f"{pair=}") for problem in problems_list for pair in problem]
        return [pair.squeeze(0) if squeeze else pair for problem in problems_list for pair in problem]

from itertools import islice
bert_tokenizer, bert_model = get_tokenizer_model()

@memory.cache
def get_problem_embeddings(problems, max_input_length, verbose=False):
    def get_paragraph_pairs(problem_text, max_input_length):
        def get_embeddings(paragraph):
            inputs = bert_tokenizer(paragraph, return_tensors='pt', padding=True, truncation=True)

            # Move inputs to GPU
            if DEVICE == "cuda":
                inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

            # Step 4: Generate embeddings
            with torch.no_grad():
                outputs = bert_model(**inputs)

            # The last hidden state contains the embeddings for each token
            return outputs.last_hidden_state
        # print(problem_text)
        paragraph_embeddings = [pad_paragraph(get_embeddings(para[:max_input_length]), max_input_length) for para in problem_text]
        # print(f"{[paras.shape for paras in paragraph_embeddings]=}")
        # print(f"{[x.shape for x in paragraph_embeddings]}")
        # print(f"{len(paragraph_embeddings)=}")
        pairs = itertools.combinations(paragraph_embeddings, 2)
        return [torch.flatten(torch.stack(pair, dim=2), start_dim=1, end_dim=2) for pair in pairs]
    if verbose:
        return [get_paragraph_pairs(problem_text, max_input_length) for problem_text in tqdm(problems)]
    else:
        return [get_paragraph_pairs(problem_text, max_input_length) for problem_text in problems]

class Pan21PyDataset(PyDataset):
    def __init__(self, x_path, y_path, max_input_length=256, batch_size=128,**kwargs):
        super().__init__(**kwargs)

        self.x_path = x_path
        y = read_ground_truth_files(y_path)
        self.task_3_y = get_task_3_ground_truth(get_simple_ground_truth(y, range(1, len(y)+1)))
        self.task_3_lens = [len(problem) for problem in self.task_3_y]
        self.num_problems = len(self.task_3_y)
        self.pair_end_index = np.cumsum(self.task_3_lens)
        self.num_pairs = np.sum(self.task_3_lens)

        self.max_input_length = max_input_length
        self.batch_size = batch_size

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.num_pairs / self.batch_size)

    def get_data(self, low_problem_idx, high_problem_idx, low_idx, high_idx):
        # print(f"{low_problem_idx=} {high_problem_idx=} {low_idx=} {high_idx=}")
        if high_problem_idx < self.num_problems:
            high_problem_idx += 1
            # print(f"{self.x_path=}")
            embeddings = get_problem_embeddings(read_problem_files(self.x_path, low_problem_idx, high_problem_idx), self.max_input_length)

            batch_x = np.array(flatten_problems(embeddings, squeeze=True)[low_idx:high_idx])
            batch_y = np.array(flatten_problems(self.task_3_y[low_problem_idx:high_problem_idx])[low_idx:high_idx])
        else:
            embeddings = get_problem_embeddings(read_problem_files(self.x_path, low_problem_idx, high_problem_idx+1), self.max_input_length)

            batch_x = np.array(flatten_problems(embeddings, squeeze=True)[low_idx:high_idx])
            batch_y = np.array(flatten_problems(self.task_3_y[low_problem_idx])[low_idx:high_idx])

        # print(f"{self.task_3_y[low_problem_idx:high_problem_idx]=}")
        return batch_x, batch_y

    def __getitem__(self, idx):
        # print(f"{idx=}")
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, self.num_pairs)
        # print(f"{low=} {high=}")

        low_problem_index = 0
        high_problem_index = 0
        low_index_within_problem = 0
        high_index_within_problem = 0
        low_found = False
        for problem_num, (end, num_pairs_in_problem) in enumerate(zip(self.pair_end_index, self.task_3_lens)):
            start = end - num_pairs_in_problem
            if not low_found:
                if end > low:
                    low_problem_index = problem_num
                    low_index_within_problem = low - start
                    low_found = True
            if end >= high:
                high_problem_index = problem_num
                high_index_within_problem = low_index_within_problem + self.batch_size
                break

        batch_x, batch_y = self.get_data(low_problem_index, high_problem_index, low_index_within_problem, high_index_within_problem)

        assert batch_x.shape[0] != 0, f"{idx=}: Dimension is 0 {low_problem_index}, {high_problem_index}, {low_index_within_problem}, {high_index_within_problem}"
        assert batch_x.shape[0] == batch_y.shape[0], f"{idx=}: Dimension mismatch {batch_x.shape=} {batch_y.shape=}"

        return batch_x, batch_y

class Pan21FourierDataset(Pan21PyDataset):
    def __init__(self, x_set, y_set, batch_size=128, num_fourier_features=512, **kwargs):
        super().__init__(x_set, y_set, batch_size=batch_size, **kwargs)
        self.num_fourier_features = num_fourier_features

    def __getitem__(self, idx):
        batch_x_embeddings, batch_y_embeddings = super().__getitem__(idx)

        if self.num_fourier_features == 0:
            pass
        else:
            num_features = batch_x_embeddings.shape[1]
            # print(f"{num_features=}")
            # 0:x will be BERT embeddings for paragraph 1
            # x:length/2 will be fourier features for paragraph 1
            num_non_fourier_features = num_features - self.num_fourier_features
            para1_fourier_features_low, para1_fourier_features_high = num_non_fourier_features // 2, num_features // 2
            para2_fourier_features_low, para2_fourier_features_high = (num_features + num_non_fourier_features) // 2, num_features

            para1_end = num_features//2
            para1_fft = np.real(np.fft.fft(batch_x_embeddings[:, :para1_end, :], axis=1))
            para2_fft = np.real(np.fft.fft(batch_x_embeddings[:, para1_end:, :], axis=1))

            # Simple decimation to fit the spectrum
            factor = num_features // self.num_fourier_features
            # print(f"{factor=} {para1_fourier_features_low}:{para1_fourier_features_high}")

            batch_x_embeddings[:, para1_fourier_features_low:para1_fourier_features_high] = para1_fft[:, ::factor, :]
            batch_x_embeddings[:, para2_fourier_features_low:para2_fourier_features_high] = para2_fft[:, ::factor, :]

        return batch_x_embeddings, batch_y_embeddings

import gc

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

from scipy.signal import butter, lfilter
# from cupyx.scipy.signal import lfilter

class Pan21FourierFilterDataset(Pan21PyDataset):
    def __init__(self, x_set, y_set,  cutoff_frequencies, **kwargs):
        super().__init__(x_set, y_set, **kwargs)
        
        order = 5
        # clamp freqencies
        cutoff_frequencies = [max(cutoff_frequencies[0], .001), min(cutoff_frequencies[1], .999)]
        
        self.i, self.u = butter(order, cutoff_frequencies, btype='bandstop')
    
    def __getitem__(self, idx):
        batch_x, batch_y = super().__getitem__(idx)

        _, j, _ = batch_x.shape

        batch_x[:, :j//2, :] = lfilter(self.i, self.u, batch_x[:, :j//2, :], axis=1)
        batch_x[:, j//2:, :] = lfilter(self.i, self.u, batch_x[:, j//2:, :], axis=1)

        return batch_x, batch_y