import torch
from pathlib import Path
# from tensorflow.python.keras.callbacks import EarlyStopping
# from tensorflow.python.keras.callbacks import ModelCheckpoint
# from tensorflow.python.keras.callbacks import History

# EarlyStopping = EarlyStopping
# History = History

# class ModelCheckpoint(ModelCheckpoint):
#     """Save the model after every epoch.

#     `filepath` can contain named formatting options,
#     which will be filled the value of `epoch` and
#     keys in `logs` (passed in `on_epoch_end`).

#     For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
#     then the model checkpoints will be saved with the epoch number and
#     the validation loss in the filename.

#     Arguments:
#         filepath: string, path to save the model file.
#         monitor: quantity to monitor.
#         verbose: verbosity mode, 0 or 1.
#         save_best_only: if `save_best_only=True`,
#             the latest best model according to
#             the quantity monitored will not be overwritten.
#         mode: one of {auto, min, max}.
#             If `save_best_only=True`, the decision
#             to overwrite the current save file is made
#             based on either the maximization or the
#             minimization of the monitored quantity. For `val_acc`,
#             this should be `max`, for `val_loss` this should
#             be `min`, etc. In `auto` mode, the direction is
#             automatically inferred from the name of the monitored quantity.
#         save_weights_only: if True, then only the model's weights will be
#             saved (`model.save_weights(filepath)`), else the full model
#             is saved (`model.save(filepath)`).
#         period: Interval (number of epochs) between checkpoints.
#     """

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         self.epochs_since_last_save += 1
#         if self.epochs_since_last_save >= self.period:
#             self.epochs_since_last_save = 0
#             filepath = self.filepath.format(epoch=epoch + 1, **logs)
#             if self.save_best_only:
#                 current = logs.get(self.monitor)
#                 if current is None:
#                     print('Can save best model only with %s available, skipping.' % self.monitor)
#                 else:
#                     if self.monitor_op(current, self.best):
#                         if self.verbose > 0:
#                             print('Epoch %05d: %s improved from %0.5f to %0.5f,'
#                                   ' saving model to %s' % (epoch + 1, self.monitor, self.best,
#                                                            current, filepath))
#                         self.best = current
#                         if self.save_weights_only:
#                             torch.save(self.model.state_dict(), filepath)
#                         else:
#                             torch.save(self.model, filepath)
#                     else:
#                         if self.verbose > 0:
#                             print('Epoch %05d: %s did not improve from %0.5f' %
#                                   (epoch + 1, self.monitor, self.best))
#             else:
#                 if self.verbose > 0:
#                     print('Epoch %05d: saving model to %s' %
#                           (epoch + 1, filepath))
#                 if self.save_weights_only:
#                     torch.save(self.model.state_dict(), filepath)
#                 else:
#                     torch.save(self.model, filepath)

# just for reference
class CustomCallback():
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

class CallbackList():
    def __init__(self, callback_list):
        self.callback_list = callback_list

    def set_model(self, model):
        self.model = model

    # for repeat use
    def set_model_for_callbacks_logs(self, logs):
        if logs is None:
            logs = {'model': self.model}
        else:
            logs['model'] = self.model
        return logs

    def on_train_begin(self, logs=None):
        if self.callback_list is not None and len(self.callback_list)>0:
            logs = self.set_model_for_callbacks_logs(logs)
            for cb in self.callback_list:
                cb.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        if self.callback_list is not None and len(self.callback_list)>0:
            logs = self.set_model_for_callbacks_logs(logs)
            for cb in self.callback_list:
                cb.on_train_end(logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        if self.callback_list is not None and len(self.callback_list)>0:
            logs = self.set_model_for_callbacks_logs(logs)
            for cb in self.callback_list:
                cb.on_epoch_begin(epoch=epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.callback_list is not None and len(self.callback_list)>0:
            logs = self.set_model_for_callbacks_logs(logs)
            for cb in self.callback_list:
                cb.on_epoch_end(epoch=epoch, logs=logs)

    # def on_test_begin(self, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_test_bigin(logs=logs)

    # def on_test_end(self, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_test_end(logs=logs)

    # def on_predict_begin(self, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_predict_bigin(logs=logs)

    # def on_predict_end(self, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_predict_end(logs=logs)

    # def on_train_batch_begin(self, batch, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_train_batch_begin(batch=batch, logs=logs)

    # def on_train_batch_end(self, batch, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_train_batch_end(batch=batch, logs=logs)

    # def on_test_batch_begin(self, batch, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_test_batch_begin(batch=batch, logs=logs)

    # def on_test_batch_end(self, batch, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_test_batch_end(batch=batch, logs=logs)

    # def on_predict_batch_begin(self, batch, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_predict_batch_begin(batch=batch, logs=logs)

    # def on_predict_batch_end(self, batch, logs=None):
    #     if self.callback_list is not None and len(self.callback_list)>0:
    #         for cb in self.callback_list:
    #             cb.on_predict_batch_end(batch=batch, logs=logs)


class EarlyStopping():
    # def __init__(self, monitor='loss',
    # min_delta=0.05,
    # patience=2,
    # verbose=0,
    # mode='auto',
    # baseline=None,
    # restore_best_weights=False)
    def __init__(self,
            monitor='val_loss',
            patience=5,
            best_metric=None,
            save_path=None,
            logger=None):
        self.monitor=monitor
        self.patience=patience
        self.patience_now = 0
        self.best_metric = best_metric
        self.save_path = save_path
        self.logger=logger
        self.higher_metric = ['auc','val_auc','acc','val_acc']
        self.lower_metric = ['binary_crossentropy','val_binary_crossentropy','logloss','val_logloss']

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(f'self.monitor: {self.monitor}, self.best_metric: {self.best_metric}')
    
        if (self.save_path is not None) and (self.best_metric is None or ((self.monitor in self.lower_metric) and (logs[self.monitor] < self.best_metric)) or ((self.monitor in self.higher_metric) and (logs[self.monitor] > self.best_metric))):
            # saving
            save_path = f'{self.save_path}_epoch_{epoch}_{self.monitor}_{logs[self.monitor]}.pt'
            save_dir = '/'.join(save_path.split('/')[:-1])
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(logs['model'].state_dict(), save_path)

            # onnx not soppose yet, because need fine tuning.
            # torch.onnx.export(logs.model,               # model being run
            #                   (X1_train, X2_train),                         # model input (or a tuple for multiple inputs)
            #                   model_save_path_tmp + ".onnx",   # where to save the model (can be a file or file-like object)
            #                   export_params=True,        # store the trained parameter weights inside the model file
            #                   opset_version=10,          # the ONNX version to export the model to
            #                   do_constant_folding=True,  # whether to execute constant folding for optimization
            #                   input_names = ['input_1','input_2'],   # the model's input names
            #                   output_names = ['output'], # the model's output names
            #                   dynamic_axes={'input_1' : {0 : 'batch_size'}, # variable length axes
            #                                 'input_2' : {0 : 'batch_size'},
            #                                 'output' : {0 : 'batch_size'}})

            self.logger.info(f'{save_path} is saved...')
        
        # first time
        if self.best_metric is None:
            self.best_metric = logs[self.monitor]
            return

        # early sotpping
        if ((self.monitor in self.lower_metric) and (logs[self.monitor] >= self.best_metric)) or ((self.monitor in self.higher_metric) and (logs[self.monitor] <= self.best_metric)):
            self.patience_now += 1
            self.logger.info(f'patience_now: {self.patience_now}')
            if self.patience_now > self.patience:
                logs['model'].stop_training = True
                return

        # update
        if ((self.monitor in self.lower_metric) and (logs[self.monitor] < self.best_metric)) or ((self.monitor in self.higher_metric) and (logs[self.monitor] > self.best_metric)):
            self.best_metric = logs[self.monitor]

        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass