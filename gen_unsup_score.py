
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model

from util.utils import set_seed
from util.utils_unsup import evaluate, export_score


def initail_setting():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    set_seed(2022)

    return opt, opt.gpu_ids[0]

def model_prediction(opt):
  data_loader = CreateDataLoader(opt)
  dataset_list = [data_loader['normal'],data_loader['smura']]
  model = create_model(opt)
  return evaluate(opt, dataset_list, model)

if __name__ == "__main__":
    opt, gpu = initail_setting()  
    res = model_prediction(opt)
    export_score(res, opt.results_dir)