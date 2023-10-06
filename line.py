import tensorflow as tf

import numpy as np
import argparse
from model import LINEModel
from utils import DBLPDataLoader
import pickle
import time
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=100)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=500)
    # 300条-2000/3000  600条-3000/4000    1000条-6000 3000条-8000/10000  5000条-10000/12000   2500条-7500(0.7/0.4)
    # 300条-xx  600条-3000    1000条-3000  3000条-6000  5000条-7000   2500条-xx(0.9)
    parser.add_argument('--total_graph', default=True)
    # parser.add_argument('--graph_file', default='data/gen-sep-train-3000-net.pkl')
    parser.add_argument('--graph_file', default='data/0_1_5850275_net.pkl')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

loss_list = []
def train(args):
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
                loss_list.append(loss)  # append the current batch loss to the loss list
            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('data/tese_%s1.pkl' % suffix, 'wb'))


def test(args):
    pass

if __name__ == '__main__':
    main()

    # plot and display the loss curve
    plt.plot(loss_list)
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

