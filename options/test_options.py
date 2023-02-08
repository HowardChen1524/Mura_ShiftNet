from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # save
        parser.add_argument('--results_dir', type=str, default='./exp_result', help='saves results here.')
        # data
        parser.add_argument('--data_version', type=str)       
        parser.add_argument('--testing_normal_dataroot', type=str, default='', help='test normal data path')
        parser.add_argument('--testing_smura_dataroot', type=str, default='', help='test smura data path') 
        parser.add_argument('--csv_path', type=str)
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--how_many', type=int, default=0, help='how many test images to run')
        parser.add_argument('--normal_how_many', type=int, default=0, help='how many test images to run')
        parser.add_argument('--smura_how_many', type=int, default=0, help='how many test images to run')
        parser.add_argument('--conf_csv_dir', type=str, help='supervised or ensemble')
        parser.add_argument('--score_csv_dir', type=str, help='using record to test')
        # model 
        parser.add_argument('--sup_model_version', type=str)
        parser.add_argument('--sup_model_path', type=str)
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load (shiftnet model)? set to latest to use latest cached model')
        # test
        parser.add_argument('--inpainting_mode', type=str, default='ShiftNet', help='[ShiftNet|OpenCV|Mean], OpenCV and Mean only for grayscale')        
        parser.add_argument('--measure_mode', type=str, default='MSE', help='[MSE|Mask_MSE|MSE_SSIM|Mask_MSE_SSIM|D_model_score|Mask_D_model_score], if need sliding add tail, e.g. MSE_sliding')
        parser.add_argument('--minmax', action='store_true', help='minmax anomaly score')
        parser.add_argument('--pos_normalize', action='store_true', help='do position normalize')
        parser.add_argument('--using_threshold', action='store_true', help='using threshold to do blind test')
        parser.add_argument('--using_record', action='store_true', help='using record to test')
        # visual position
        parser.add_argument('--binary_threshold', type=float, help='using record to test')
        parser.add_argument('--min_area', type=int, help='using record to test')

        self.isTrain = False

        return parser

# parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
# parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
# parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
# parser.add_argument('--testing_mask_folder', type=str, default='masks/testing_masks', help='perpared masks for testing')
