import time
from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from data import dataset_type
from data.dataset import Dataset
from layers.basics import optimize
from models import model_type
from models.model_type import MODELS
from utils.batch_helper import BatchHelper
from utils.config_helpers import MainConfig
from utils.data_utils import DatasetVectorizer
from utils.data_utilsMB import DatasetVectorizerMB
from utils.log_saver import LogSaver
from utils.model_evaluator import ModelEvaluator
from utils.model_saver import ModelSaver
from models.MemoryBank_crest import Memorybank
from utils.other_utils import timer, set_visible_gpu, init_config
from sklearn.metrics import f1_score
from models.SPLD import spld
import csv
log = tf.logging.info


def create_experiment_name(model_name, main_config, model_config):
    experiment_name = '{}_{}'.format(model_name, main_config['PARAMS']['embedding_size'])
    if model_name == model_type.ModelType.rnn.name:
        experiment_name += ("_" + model_config['PARAMS']['cell_type'])
    
    experiment_name += ("_" + main_config['PARAMS']['loss_function'])
    
    return experiment_name


def train(
        main_config,
        model_config,
        model_name,
        experiment_name,
        dataset_name,
):
    trainf1 = []
    devf1 = []
    tloss = []
    x = []
    spllosss = []

    selected_idx_list = []
    # 自步学习超参数
    lam = 2
    gamma = 1.5

    # 自步学习参数的学习了调整参数
    u1 = 2
    u2 = 2

    main_cfg = MainConfig(main_config)
    # get instances from config
    model = MODELS[model_name]
    dataset = dataset_type.get_dataset(dataset_name)
    
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=main_cfg.log_device_placement,
    )
    # train data get
    train_data = dataset.train_set_pairs()
    vectorizer = DatasetVectorizer(
        model_dir=main_cfg.model_dir,
        char_embeddings=main_cfg.char_embeddings,
        raw_sentence_pairs=train_data,
    )
    
    dataset_helper = Dataset(vectorizer, dataset, main_cfg.batch_size)
    max_sentence_len = vectorizer.max_sentence_len
    vocabulary_size = vectorizer.vocabulary_size
    
    # train sentence
    train_mini_sen1, train_mini_sen2, train_mini_labels = dataset_helper.pick_train_mini_batch()
    train_mini_labels = train_mini_labels.reshape(-1, 1)
    
    # dev sentences
    dev_mini_sen1, dev_mini_sen2, dev_mini_labels = dataset_helper.pick_dev_mini_batch()
    dev_mini_labels = dev_mini_labels.reshape(-1, 1)

    # test sentence
    test_sentence1, test_sentence2 = dataset_helper.test_instances()
    test_labels = dataset_helper.test_labels()
    test_labels = test_labels.reshape(-1, 1)
    
    num_batches = dataset_helper.num_batches
    
    session = tf.Session(config=config)
    # get model
    model = model(
        max_sentence_len,
        vocabulary_size,
        main_config,
        model_config,
    )

    # save model
    model_saver = ModelSaver(
        model_dir=main_cfg.model_dir,
        model_name=experiment_name,
        checkpoints_to_keep=main_cfg.checkpoints_to_keep,
    )
    
    v_star = np.random.randint(0.0, 2.0, size=(num_batches, main_config['TRAINING'].getint('batch_size')))
    #samples_label_arr = np.array([0.0,1.0])
    x = range(0,main_cfg.num_epochs)
    #lr = main_cfg['TRAINING'].getfloat('learning_rate') # get learning rate from config
    # training 

    writer=tf.summary.FileWriter('../log_dir/tensorboard_study', session.graph)
    global_step = 0
    # initializer
    init = tf.global_variables_initializer()
    session.run(init)
    
    log_saver = LogSaver(
        main_cfg.logs_path,
        experiment_name,
        dataset_name,
        session.graph,
    )
    model_evaluator = ModelEvaluator(model, session)
    
    metrics = {'acc': 0.0}
    time_per_epoch = []
    train_loss = 0.0
    log('Training model for {} epochs'.format(main_cfg.num_epochs))
    for epoch in tqdm(range(main_cfg.num_epochs), desc='Epochs'):
        
        start_time = time.time()
        
        train_sentence1, train_sentence2 = dataset_helper.train_instances(shuffle=True)
        # print("train sentence dim is ",train_sentence1.shape)
        train_labels = dataset_helper.train_labels()
        
        train_batch_helper = BatchHelper(
            train_sentence1,
            train_sentence2,
            train_labels,
            main_cfg.batch_size,
        )
        
        # small eval set for measuring dev accuracy
        dev_sentence1, dev_sentence2, dev_labels = dataset_helper.dev_instances()
        dev_labels = dev_labels.reshape(-1, 1)
        
        tqdm_iter = tqdm(range(num_batches), total=num_batches, desc="Batches", leave=False,
                            postfix=metrics)

        for batch in tqdm_iter:
            global_step += 1
            sentence1_batch, sentence2_batch, labels_batch = train_batch_helper.next(batch)
            #print(labels_batch.shape)
            v_star_feed = v_star[batch]
            v_star_feed = np.expand_dims(v_star_feed,axis=0)
            #print("batch is:", batch)
            labels = np.squeeze(labels_batch,axis = 1)
            idx0_for_each_group = np.where(labels == 0)[0]
            #print(idx0_for_each_group)
            idx1_for_each_group = np.where(labels == 1)[0]
            feed_dict_train_spl = {
                model.x1: sentence1_batch,
                model.x2: sentence2_batch,
                model.is_training: True,
                model.labels: labels_batch,
                model.v_star:v_star_feed,
                model.lamda:lam,
                model.gamma:gamma,
                model.idx0:idx0_for_each_group,
                model.idx1:idx1_for_each_group,
            }
            loss, splloss,  _ = session.run([model.spldloss, model.splloss, model.opt], feed_dict=feed_dict_train_spl)
            
            feed_dict = {
                model.x1: sentence1_batch,
                model.x2: sentence2_batch,
                model.is_training: True,
                model.labels: labels_batch,
            }

            train_loss = train_loss + loss.mean().item()
            new_loss_T = session.run([model.spldloss],feed_dict = feed_dict)
            new_loss_T = np.array(new_loss_T)

            labels_batch_list = []
            new_loss = []
            for i in range(len(labels_batch)):
                labels_batch_list.append(labels_batch[i,0])
                new_loss.append(new_loss_T[0,i])
    
            selected_idx_arr = spld(new_loss,labels_batch_list,lam,gamma)
            selected_idx_list.append(selected_idx_arr)

            # 验证模型
            if batch % main_cfg.eval_every == 0:
                labels = np.squeeze(train_mini_labels,axis = 1)
                idx0_for_each_group = np.where(labels == 0)[0]
                # print(idx0_for_each_group)
                idx1_for_each_group = np.where(labels == 1)[0]
                feed_dict_train = {
                    model.x1: train_mini_sen1,
                    model.x2: train_mini_sen2,
                    model.is_training: False,
                    model.labels: train_mini_labels,
                    model.v_star:v_star_feed,
                    model.lamda:lam,
                    model.gamma:gamma,
                    model.idx0:idx0_for_each_group,
                    model.idx1:idx1_for_each_group,
                }
                
                train_accuracy, train_f1, train_summary = session.run(
                    [model.accuracy, model.f1, model.summary_op],
                    feed_dict=feed_dict_train,
                )
                log_saver.log_train(train_summary, global_step)

                dev_labels_eval = np.squeeze(dev_mini_labels,axis = 1)
                idx0_for_each_group = np.where(dev_labels_eval == 0)[0]
                #print(idx0_for_each_group)
                idx1_for_each_group = np.where(dev_labels_eval == 1)[0]
                feed_dict_dev = {
                    model.x1: dev_mini_sen1,
                    model.x2: dev_mini_sen2,
                    model.is_training: False,
                    model.labels: dev_mini_labels,
                    model.v_star:v_star_feed,
                    model.lamda:lam,
                    model.gamma:gamma,
                    model.idx0:idx0_for_each_group,
                    model.idx1:idx1_for_each_group,
                }
                dev_accuracy, dev_f1, dev_summary = session.run(
                    [model.accuracy,model.f1, model.summary_op],
                    feed_dict=feed_dict_dev,
                )
                log_saver.log_dev(dev_summary, global_step)

                tqdm_iter.set_postfix(
                    train_acc='{:.2f}'.format(float(train_accuracy)),
                    loss='{:.2f}'.format(float(train_loss)),
                    splloss='{:.2f}'.format(float(splloss)),
                    train_f1 = '{:.2f}'.format(float(train_f1)),
                    dev_f1 = '{:.2f}'.format(float(dev_f1)),
                    dev_acc = '{:.2f}'.format(float(dev_accuracy)),
                    epoch=epoch
                )
                tloss.append(train_loss)
                spllosss.append(splloss.item())
                train_loss = 0.0
            v_star[batch] = np.zeros((len(new_loss),),dtype=np.float32)

            for selected_idx in selected_idx_arr:
                v_star[batch][selected_idx] = 1
                
            if global_step % main_cfg.save_every == 0:
                model_saver.save(session, global_step=global_step)
        lam = u1 * lam
        gamma = u2 * gamma

        model_evaluator.evaluate_dev(
            x1=dev_sentence1,
            x2=dev_sentence2,
            labels=dev_labels,
        )
        
        end_time = time.time()
        total_time = timer(start_time, end_time)
        time_per_epoch.append(total_time)
        
        model_saver.save(session, global_step=global_step)
        print('\ndev_f1 = {}, train_f1 = {}, dev_acc = {}, train_acc = {}, loss = {} '.format(dev_f1, train_f1, dev_accuracy, train_accuracy, train_loss))
        trainf1.append(train_f1)
        devf1.append(dev_f1)
    model_evaluator.evaluate_test(test_sentence1, test_sentence2, test_labels)
    model_evaluator.save_evaluation(
        model_path='{}/{}'.format(
            main_cfg.model_dir,
            experiment_name,
        ),
        epoch_time=time_per_epoch[-1],
        dataset=dataset,
    )
    #visualization(trainf1,devf1,tloss,splloss,x)
    writer.close()
    

def visualization(train_f1,dev_f1,loss, splloss, x):
    plt.subplot(2,2,1)
    plt.plot(x,train_f1,color= 'r')
    plt.title('The curve of train_f1')

    plt.subplot(2,2,2)
    plt.plot(x,dev_f1,color= 'g')
    plt.title('The curve of dev_f1')

    plt.subplot(2,2,3)
    plt.plot(len(loss),loss,color= 'r')
    plt.title('The curve of loss')

    plt.subplot(2,2,4)
    plt.plot(len(splloss),splloss,color= 'r')
    plt.title('The curve of splloss')

    plt.savefig('train.jpg')


def predict(
        main_config,
        model_config,
        model,
        experiment_name,
):
    model = MODELS[model]
    main_cfg = MainConfig(main_config)
    # model_dir = str(main_config['DATA']['model_dir'])
    
    vectorizer = DatasetVectorizer(
        model_dir=main_cfg.model_dir,
        char_embeddings=main_cfg.char_embeddings,
    )
    
    max_doc_len = vectorizer.max_sentence_len
    vocabulary_size = vectorizer.vocabulary_size
    
    model = model(max_doc_len, vocabulary_size, main_config, model_config)
    
    with tf.Session() as session:
        saver = tf.train.Saver()
        last_checkpoint = tf.train.latest_checkpoint(
            '{}/{}'.format(
                main_cfg.model_dir,
                experiment_name,
            )
        )
        saver.restore(session, last_checkpoint)
        while True:
            x1 = input('First sentence:')
            x2 = input('Second sentence:')
            x1_sen = vectorizer.vectorize(x1)
            x2_sen = vectorizer.vectorize(x2)
            
            feed_dict = {model.x1: x1_sen, model.x2: x2_sen, model.is_training: False}
            prediction = session.run([model.temp_sim], feed_dict=feed_dict)
            print(prediction)

def main():
    parser = ArgumentParser()
    
    parser.add_argument(
        '--mode',
        default='train',
        #choices=['train', 'predict'],
        help='pipeline mode',
    )
    
    parser.add_argument(
        '--model',
        default='bilstm_mhatt',
        #choices=['rnn', 'cnn', 'multihead','bilstm_mhatt'],
        help='model to be used',
    )
    
    parser.add_argument(
        '--dataset',
        default='Crest',
        #choices=['Crest','ECSIN','ECMUL','SCISIN','SCIMUL'],
        nargs='?',
        help='dataset to be used',
    )
    
    parser.add_argument(
        '--experiment_name',
        required=False,
        help='the name of run experiment',
    )
    
    parser.add_argument(
        '--gpu',
        default='0',
        help='index of GPU to be used (default: %(default))',
    )
    
    args = parser.parse_args()
    if 'train' in args.mode:
        if args.dataset is None:
            parser.error('Positional argument [dataset] is mandatory')
    set_visible_gpu(args.gpu)
    
    main_config = init_config()
    model_config = init_config(args.model)
    print("Load model config!!!")
    print(model_config)
    
    mode = args.mode
    
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = create_experiment_name(args.model, main_config, model_config)

    sentences = []
    labels = []
    directions = []
    span1s = []
    span2s = []
    fr = open('corpora/Crest/crest.csv','r',encoding='utf-8')

    reader = csv.reader(fr)

    for line in reader:
        sentences.append(line[6])
        labels.append(line[8])
        span1 = line[3].strip('[\'').strip('\']')
        span2 = line[4].strip('[\'').strip('\']')
        span1s.append(span1)
        span2s.append(span2)
        directions.append(line[9])
    fr.close()

    main_cfg = MainConfig(main_config)
    # train data get
    train_data = sentences
    vectorizer = DatasetVectorizerMB(
        model_dir=main_cfg.model_dir,
        char_embeddings=main_cfg.char_embeddings,
        raw_sentence=train_data,
    )
    MB = Memorybank(vectorizer,sentences,labels,span1s,span2s,directions)
    MB.Tvector()

    if 'train' in mode:
        train(main_config, model_config, args.model, experiment_name, args.dataset)
    else:
        predict(main_config, model_config, args.model, experiment_name)

if __name__ == '__main__':
    main()
