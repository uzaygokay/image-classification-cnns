from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning import Trainer

from custom_logger import get_custom_logger
from arguments import get_args
from datamodule import CIFAR10_dm
from model import ConvModel


if __name__ == '__main__':

    args = get_args()
    logger = get_custom_logger(__name__)

    logger.info('Init Datamodule')
    dm = CIFAR10_dm(
        data_dir=args.data_dir,
        batch_size= args.batch_size,
        num_workers=args.num_workers
    )

    logger.info('Init Model')
    model = ConvModel(
        *dm.data_dims,
        num_classes=dm.num_classes,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout
    )

    logger.info('Init Callbacks')
    callbacks = []
    callbacks.append(RichProgressBar(refresh_rate=200, leave=True))


    logger.info('Init Trainer')
    trainer = Trainer(
        fast_dev_run = 3,
        max_epochs = args.max_epochs,
        enable_progress_bar=True,
        callbacks=callbacks,
        limit_train_batches =0.1,
        limit_val_batches = 0.1
    )

    logger.info('Run Trainer')
    trainer.fit(model,dm)

    #logger.info('Test the best model')
    #trainer.test(ckpt_path='best', datamodule=dm)