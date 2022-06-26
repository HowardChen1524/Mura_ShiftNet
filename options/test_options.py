from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # parser.add_argument('--which_epoch', type=str, default='20', help='which epoch to load? set to latest to use latest cached model')\
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=1000, help='how many test images to run')
        parser.add_argument('--testing_mask_folder', type=str, default='masks/testing_masks', help='perpared masks for testing')
        # Howard add
        parser.add_argument('--testing_normal_dataroot', type=str, default='', help='test normal data path')
        parser.add_argument('--testing_smura_dataroot', type=str, default='', help='test smura data path')        
        parser.add_argument('--normal_how_many', type=int, default=1000, help='how many test images to run')
        parser.add_argument('--smura_how_many', type=int, default=1000, help='how many test images to run')
        parser.add_argument('--inpainting_mode', type=str, default='ShiftNet', help='[ShiftNet|OpenCV|Mean]')        
        parser.add_argument('--measure_mode', type=str, default='MSE', help='[MAE|MSE|Mask_MAE|Mask_MSE|Discriminator|Mask_Discriminator]')

        self.isTrain = False

        return parser
