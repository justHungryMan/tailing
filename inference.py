import torch
import theconf
from theconf import Config as C
import logging
import sys

import trainer
from train import evaluation

def infer(flags):
    device = torch.device(type= 'cuda', index=0)

    model = trainer.model.create(C.get()['architecture'])
    model.to(device=device, non_blocking=True)
   
    logging.info(f"[Model] | Load from {C.get()['inference']['checkpoint']}")

    try:
        checkpoint = torch.load(C.get()['inference']['checkpoint'], map_location=device)['model']
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        model.load_state_dict(checkpoint)
    except Exception as e:
        raise (e)



    test_loader, _ = trainer.dataset.create(C.get()['dataset'],
                                                    mode='test')
    logging.info(f'[Dataset] | test_examples: {len(test_loader)}')

    criterion = trainer.loss.create(C.get()['loss']).to(device=device, non_blocking=True)

    model.eval()    
    test_loss, test_acc, report = evaluation(0, model, test_loader, criterion, device, flags)
   


if __name__ == '__main__':
    parser = theconf.ConfigArgumentParser(conflict_handler='resolve')
    flags = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)
    
    flags.is_master = True
    infer(flags)