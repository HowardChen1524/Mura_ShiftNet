def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'shiftnet':
        from models.shift_net.shiftnet_model import ShiftNetModel
        model = ShiftNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    # call initialize
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
