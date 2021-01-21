import os
import skimage
import skimage.io
import skimage.transform
import utils
import time
import glob

def save_train_info(num_exp, seperate_by, train_type, epochs, batch_size, leaning_rate,
                    n_train, n_val, device_type):

    now = time.localtime()
    now_t = "%04d_%02d_%02d %02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    f = open(f'./result/{num_exp}/train_info.txt', 'w')
    data = f'''training info
Experience number:      {num_exp}
Seperate type:          {seperate_by}
Train type:             {train_type}
Epoch:                  {epochs}
Batch size:             {batch_size}
Learning rate:          {leaning_rate}
Training size:          {n_train}
Validation size:        {n_val}
Device:                 {device_type}
Local time:             {now_t}


Epoch / Accuracy / Loss
'''
    f.write(data)
    f.close()

    return now_t

def save_feature_info(num_exp, add_percent,
                        feature_epoch_iter, how_many_times_add_feature, learning_rate_feature, device_type):

    now = time.localtime()
    now_t = "%04d_%02d_%02d %02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    f = open(f'./result/{num_exp}/feature_train_info.txt', 'w')
    data = f'''feature training info
Experience number:      {num_exp}
Add persent:            {add_percent}
Feature epoch:          {feature_epoch_iter}
Times add feature:      {how_many_times_add_feature}
Feature learning rate:  {learning_rate_feature}
Device:                 {device_type}
Local time:             {now_t}


Epoch / Accuracy / Loss
'''
    f.write(data)
    f.close()

    return now_t

def save_test_info(num_exp, n_test):

    now = time.localtime()
    now_t = "%04d_%02d_%02d %02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    f = open(f'./result/{num_exp}/test_info.txt', 'w')
    data = f'''test info
Experience number:      {num_exp}
Test size:              {n_test}
Local time:             {now_t}


Accuracy / Loss / Precision / Recall / Specificity / TP / TN / FP / FN
'''
    f.write(data)
    f.close()

    return now_t

def save_intermediate_train(epoch, num_exp, accuracy, loss):
    with open(f'./result/{num_exp}/train_info.txt', 'a') as f:
        f.write(f'''\n
{epoch} / {accuracy} / {loss}
''')
        f.close()

def save_intermediate_val(epoch, num_exp, accuracy,
                        loss, precision, recall, spec,
                        TP, TN, FP, FN):
    if epoch == 1:
        f = open(f'./result/{num_exp}/val_info.txt', 'w')
        data = f'''validation info
Epoch / Accuracy / Loss / Precision / Recall / Specificity / TP / TN / FP / FN
'''
        f.write(data)
        f.close()
    with open(f'./result/{num_exp}/val_info.txt', 'a') as f:
        f.write(f'''\n
{epoch} / {accuracy} / {loss} / {precision} / {recall} / {spec} / {TP} / {TN} / {FP} / {FN}
''')
        f.close()

def save_intermediate_feature_train(epoch, num_exp, accuracy,
                        loss, is_resampling, len_uf):
    if is_resampling:
        with open(f'./result/{num_exp}/feature_train_info.txt', 'a') as f:
            f.write(f'''\n
Ulcer Sample : {len_uf}
''')
            f.close()
    else:
        with open(f'./result/{num_exp}/feature_train_info.txt', 'a') as f:
            f.write(f'''\n
{epoch} / {accuracy} / {loss}
''')
            f.close()

def save_intermediate_feature_val(epoch, num_exp, accuracy,
                        loss, precision, recall, spec,
                        TP, TN, FP, FN):
    if epoch == 1:
        f = open(f'./result/{num_exp}/feature_val_info.txt', 'w')
        data = f'''feature validation info
Epoch / Accuracy / Loss / Precision / Recall / Specificity / TP / TN / FP / FN
'''
        f.write(data)
        f.close()
    with open(f'./result/{num_exp}/feature_val_info.txt', 'a') as f:
        f.write(f'''\n
{epoch} / {accuracy} / {loss} / {precision} / {recall} / {spec} / {TP} / {TN} / {FP} / {FN}
''')
        f.close()

def save_intermediate_test(num_exp, accuracy,
                        loss, precision, recall, spec,
                        TP, TN, FP, FN, dic_path, is_fm):
    if is_fm:
        file_name = 'feature_test_info.txt'
    else:
        file_name = 'test_info.txt'
    with open(f'./result/{num_exp}/{dic_path}/{file_name}', 'a') as f:
        f.write(f'''
{accuracy} / {loss} / {precision} / {recall} / {spec} / {TP} / {TN} / {FP} / {FN}
''')
        f.close()