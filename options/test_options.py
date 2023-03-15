from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # save path
        parser.add_argument('--results_dir', type=str, default='./exp_result', help='saves results here.')
        # data
        parser.add_argument('--data_version', type=str)       
        parser.add_argument('--data_csv_path', type=str)
        parser.add_argument('--sup_dataroot', type=str)
        parser.add_argument('--testing_normal_dataroot', type=str, default='', help='test normal data path')
        parser.add_argument('--testing_smura_dataroot', type=str, default='', help='test smura data path') 
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
        parser.add_argument('--measure_mode', type=str)
        parser.add_argument('--pos_normalize', action='store_true', help='do position normalize')
        parser.add_argument('--using_threshold', action='store_true', help='using threshold to do blind test')
        parser.add_argument('--using_record', action='store_true', help='using record to test')
        # visual position
        parser.add_argument('--binary_threshold', type=float, help='patch combine th')
        parser.add_argument('--min_area', type=int, help='patch combine filter min area')
        parser.add_argument('--sup_gradcam_th', type=float, help='')

        self.isTrain = False

        return parser