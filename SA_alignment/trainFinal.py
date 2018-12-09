from attentionNet import *
from DataLoader import *
from Utils import *
import scipy.io
import numpy as np
from skimage import color
from tensorflow.contrib import slim
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
os.environ['CUDA_VISIBLE_DEVICES']='0'



class Train():
    def __init__(self,class_num,batch_size,iters,learning_rate,keep_prob,param):
        self.ClassNum=class_num

        self.BatchSize=batch_size
        self.Iters=iters
        self.LearningRate=learning_rate
        self.KeepProb=keep_prob
        self.target_loss_param=param[0]
        self.domain_loss_param=param[1]
        self.adver_loss_param=param[2]

        self.SourceData,self.SourceLabel=load_svhn('svhn')
        self.SourceData2,self.SourceLabel2=load_fakemnist('synthetic')
        # # self.SourceData=np.vstack((self.SourceData1,self.SourceData2))
        # # self.SourceLabel=np.hstack((self.SourceLabel1,self.SourceLabel2))
        # # self.SourceData,self.SourceLabel=init_shuffle(self.SourceData,self.SourceLabel)
        self.TargetData, self.TargetLabel=load_mnist('mnist')
        self.TestData, self.TestLabel = load_mnist('mnist',split='test')
        ######################################################################################

        # self.SourceData,self.SourceLabel=load_realsvhn('s_train')
        # self.SourceData2,self.SourceLabel2=load_fakemnist('s_train')
        #
        # self.TargetData, self.TargetLabel=load_realmnist('s_train')
        # self.TestData, self.TestLabel = load_testrealmnist('s_train')


        #######################################################################################
        self.source_image = tf.placeholder(tf.float32, shape=[None, 32,32,1],name="source_image")
        self.source_label = tf.placeholder(tf.float32, shape=[None, self.ClassNum],name="source_label")
        #
        # self.mid_image = tf.placeholder(tf.float32, shape=[self.BatchSize, 32,32,3],name="source2_image")
        # self.mid_label = tf.placeholder(tf.float32, shape=[self.BatchSize, self.ClassNum],name="source2_label")

        self.target_image = tf.placeholder(tf.float32, shape=[None, 32, 32,1],name="target_image")
        self.Training_flag = tf.placeholder(tf.bool, shape=None,name="Training_flag")



    def TrainNet(self):
        self.source_model=Lenet(inputs=self.source_image,training_flag=self.Training_flag, reuse=False)
        self.target_model=Lenet(inputs=self.target_image, training_flag=self.Training_flag,reuse=True)
        # self.mid_model=Lenet(inputs=self.mid_image, training_flag=self.Training_flag,reuse=True)
        self.CalLoss()
        varall=tf.trainable_variables()

        self.solver = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(self.loss)
        self.source_prediction = tf.argmax(self.source_model.softmax_output, 1)
        self.target_prediction = tf.argmax(self.target_model.softmax_output, 1)

        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # self.solver = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(self.loss)
            init = tf.global_variables_initializer()
            sess.run(init)
            self.SourceLabel=sess.run(tf.one_hot(self.SourceLabel,10))
            self.TestLabel=sess.run(tf.one_hot(self.TestLabel,10))

            self.SourceLabel2 = sess.run(tf.one_hot(self.SourceLabel2, 10))
            # self.source_model.weights_initial(sess)
            # self.target_model.weights_initial(sess)
            true_num = 0.0
            lossData=[]
            for step in range(self.Iters):
                # self.SourceData,self.SourceLabel=shuffle(self.SourceData,self.SourceLabel)
                i= step % int(self.SourceData.shape[0]/self.BatchSize)
                j= step % int(self.TargetData.shape[0]/self.BatchSize)
                k= step % int(self.SourceData2.shape[0]/self.BatchSize)
                source_batch_x = self.SourceData[i * self.BatchSize: (i + 1) * self.BatchSize]
                source_batch_y = self.SourceLabel[i * self.BatchSize: (i + 1) * self.BatchSize]


                mid_batch_x = self.SourceData2[k * self.BatchSize: (k + 1) * self.BatchSize]
                mid_batch_y = self.SourceLabel2[k * self.BatchSize: (k + 1) * self.BatchSize]

                train_batch_x=np.concatenate((source_batch_x,mid_batch_x),axis=0)
                train_batch_y=np.concatenate((source_batch_y,mid_batch_y),axis=0)
                target_batch_x = self.TargetData[j * self.BatchSize: (j + 1) * self.BatchSize]
                total_loss, source_loss, attention_loss,source_prediction,_= sess.run(
                    fetches=[self.loss, self.source_loss, self.attention,self.source_prediction, self.solver],
                    feed_dict={self.source_image: train_batch_x, self.source_label: train_batch_y,self.target_image: target_batch_x, self.Training_flag: True})

                true_label = argmax(train_batch_y, 1)
                true_num = true_num + sum(true_label == source_prediction)

                # if step % 100==0:
                #     self.SourceData, self.SourceLabel = shuffle(self.SourceData, self.SourceLabel)
                if step % 200 ==0:
                    print "Iters-{} ### TotalLoss={} ### SourceLoss={} ###AttentionLoss={}".format(step, total_loss, source_loss,attention_loss)
                    train_accuracy = true_num / (200*self.BatchSize*2)
                    true_num = 0.0
                    print " ########## train_accuracy={} ###########".format(train_accuracy)
                    self.Test(sess,lossData)
                if step % 2000 == 0:
                    pass
                    # savedata = np.array(lossData)
                    # np.save("SVHNtoMNIST.npy", savedata)
                    # pass
                    # self.conputeTSNE(step, self.SourceData,  self.TargetData,self.SourceData2,self.SourceLabel, self.TargetLabel,self.SourceLabel2, sess)

            savedata=np.array(lossData)
            np.save("selfunl.npy",savedata)




    def CalLoss(self):
        fc5=self.source_model.fc5
        self.source_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.source_label[0:self.BatchSize], logits=fc5[0:self.BatchSize])
        # self.source_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.source_label, logits=fc5)
        self.mid_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.source_label[self.BatchSize:], logits=fc5[self.BatchSize:])
        self.mid_loss = tf.reduce_mean(self.mid_cross_entropy)

        self.source_loss = tf.reduce_mean(self.source_cross_entropy)
        self.attention_loss2()
        # self.CalTargetLoss(method="Entropy")
        self.CalDomainLoss(method="CORAL")
        # self.CalAdver()
        # self.L2Loss()
        self.loss=self.source_loss+self.domain_loss_param*self.domain_loss+0.5*self.attention

        # self.loss = self.source_loss+0*self.mid_loss
    def L2Loss(self):
        all_variables = tf.trainable_variables()
        self.l2 = 1e-5 * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])

    def CalDomainLoss(self,method):
        if method=="MMD":
            Xs=self.source_model.fc4
            Xt=self.target_model.fc4
            diff=tf.reduce_mean(Xs, 0, keep_dims=False) - tf.reduce_mean(Xt, 0, keep_dims=False)
            self.domain_loss=tf.reduce_sum(tf.multiply(diff,diff))


        elif method=="KMMD":
            Xs=self.source_model.fc4
            Xt=self.target_model.fc4
            self.domain_loss=tf.maximum(0.0001,KMMD(Xs,Xt))



        elif method=="LCORAL":
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            # d=int(Xs.shape[1])
            # Xms = Xs - tf.reduce_mean(Xs, 0, keep_dims=True)
            # Xcs = tf.matmul(tf.transpose(Xms), Xms) / self.BatchSize
            # Xmt = Xt - tf.reduce_mean(Xt, 0, keep_dims=True)
            # Xct = tf.matmul(tf.transpose(Xmt), Xmt) / self.BatchSize
            # self.domain_loss = tf.reduce_sum(tf.multiply((Xcs - Xct), (Xcs - Xct)))
            # self.domain_loss=self.domain_loss / (4.0*d*d)
            self.domain_loss=self.coral_loss(Xs,Xt)


        elif method =='CORAL':
            fc4=self.source_model.fc4
            Xs = fc4[0:self.BatchSize]
            Xt = self.target_model.fc4
            Xs1 = fc4[self.BatchSize:]
            self.domain_loss=self.log_coral_loss(Xs,Xt)
            self.domain_loss+=self.log_coral_loss(Xs1,Xt)
            #
            # Xt1 = self.mid_model.fc4
            # self.domain_loss=self.log_coral_loss(Xs,Xt)
            # self.domain_loss+=self.log_coral_loss(Xs,Xt1)
    def CalTargetLoss(self,method):
        if method=="Entropy":
            trg_softmax=self.target_model.softmax_output
            self.target_loss=-tf.reduce_mean(tf.reduce_sum(trg_softmax * tf.log(trg_softmax), axis=1))


        elif method=="Manifold":
            pass
    def attention_loss(self):
        self.attention=tf.reduce_mean(tf.abs(self.mid_model.cmp-self.source_model.cmp))

    def attention_loss2(self):
        att0= self.source_model.cmp
        att = self.source_model.att_flat
        att_s = att[0:self.BatchSize]
        att_t = att[self.BatchSize:]
        # att1=tf.norm(att_s,axis=1)
        self.attention=tf.norm(att_s/tf.norm(att_s,axis=1,ord='euclidean',keep_dims=True)-att_t/tf.norm(att_t,axis=1,ord='euclidean',keep_dims=True),ord='euclidean')/self.BatchSize

    def attention_lossL2(self):
        att = self.source_model.cmp
        att_s = att[0:self.BatchSize]
        att_t = att[self.BatchSize:]
        channels=np.shape(att)[3]
        attL2=np.zero
        for i in (0,np.shape(att)[3]):
            pass



    def coral_loss(self, h_src, h_trg, gamma=1e-3):

        # regularized covariances (D-Coral is not regularized actually..)
        # First: subtract the mean from the data matrix
        batch_size = self.BatchSize
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # Returns the Frobenius norm (there is an extra 1/4 in D-Coral actually)
        # The reduce_mean account for the factor 1/d^2
        return tf.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))

    def log_coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances result in inf or nan
        # First: subtract the mean from the data matrix
        batch_size = float(self.BatchSize)
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                         transpose_a=True)   #+ gamma * tf.eye(64)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                         transpose_a=True)   #+ gamma * tf.eye(64)
        # eigen decomposition
        eig_source = tf.self_adjoint_eig(cov_source)
        eig_target = tf.self_adjoint_eig(cov_target)
        log_cov_source = tf.matmul(eig_source[1],
                                   tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True))
        log_cov_target = tf.matmul(eig_target[1],
                                   tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True))

        # Returns the Frobenius norm
        return tf.reduce_mean(tf.square(tf.subtract(log_cov_source, log_cov_target)))

    # ~ return tf.reduce_mean(tf.reduce_max(eig_target[0]))
    # ~ return tf.to_float(tf.equal(tf.count_nonzero(h_src), tf.count_nonzero(h_src)))


    def Test(self,sess,lossData):
        true_num=0.0
        # num=int(self.TargetData.shape[0]/self.BatchSize)
        num = int(self.TestData.shape[0] / self.BatchSize)
        total_num=num*self.BatchSize
        for i in range (num):
            # self.TestData, self.TestLabel = shuffle(self.TestData, self.TestLabel)
            k = i % int(self.TestData.shape[0] / self.BatchSize)
            target_batch_x = self.TestData[k * self.BatchSize: (k + 1) * self.BatchSize]
            target_batch_y= self.TestLabel[k * self.BatchSize: (k + 1) * self.BatchSize]
            prediction=sess.run(fetches=self.target_prediction, feed_dict={self.target_image:target_batch_x, self.Training_flag: False})
            true_label = argmax(target_batch_y, 1)

            true_num+=sum(true_label==prediction)
        accuracy=true_num / total_num
        print "###########  Test Accuracy={} ##########".format(accuracy)
        lossData.append(accuracy)
    def conputeTSNE(self,step,source_images, target_images,mid_images,source_labels,target_labels,mid_labels,sess):

        target_images = target_images[:2000]
        target_labels = target_labels[:2000]
        source_images = source_images[:2000]
        source_labels = source_labels[:2000]
        mid_images = mid_images[:2000]
        mid_labels = mid_labels[:2000]

        target_labels = one_hot(target_labels.astype(int), 10)
        print(source_labels.shape)

        assert len(target_labels) == len(source_labels)



        n_slices = int(2000 / 128)

        fx_src = np.empty((0, 64))
        fx_trg = np.empty((0, 64))

        for src_im, trg_im,mid_im in zip(np.array_split(source_images, n_slices),
                                  np.array_split(target_images, n_slices),
                                  np.array_split(mid_images, n_slices)
                                  ):
            ss=np.concatenate((src_im,mid_im),axis=0)
            feed_dict = {self.source_image: ss, self.target_image: trg_im,self.Training_flag:False}

            fx_src_, fx_trg_ = sess.run([self.source_model.fc4, self.target_model.fc4], feed_dict)

            fx_src = np.vstack((fx_src, np.squeeze(fx_src_)))
            fx_trg = np.vstack((fx_trg, np.squeeze(fx_trg_)))

        src_labels = np.argmax(source_labels, 1)
        trg_labels = np.argmax(target_labels, 1)
        mid_labels = np.argmax(mid_labels, 1)
        # assert len(src_labels) == len(fx_src)
        assert len(trg_labels) == len(fx_trg)

        print 'Computing T-SNE.'

        model = TSNE(n_components=2, random_state=0)
        print(plt.style.available)
        plt.style.use('seaborn-paper')

        TSNE_hA = model.fit_transform(np.vstack((fx_src, fx_trg)))
        plt.figure(1,facecolor="white")
        plt.cla()
        plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels,mid_labels, trg_labels,)),s=10, cmap = mpl.cm.jet)
        plt.savefig('imgunlo/%d_h.eps'%step,format="eps",dpi=1000,bbox_inches="tight")
        plt.figure(2,facecolor="white")
        plt.cla()
        plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((2000,)),2*np.ones((2000,)), 3*np.ones((2000,)))),s=10, cmap = mpl.cm.jet)
        plt.savefig('imgunlo/%d_c.eps'%step,format="eps",dpi=1000,bbox_inches="tight")

def main():
    target_loss_param =0
    domain_loss_param =7
    adver_loss_param=0
    param=[target_loss_param, domain_loss_param,adver_loss_param]
    Runer=Train(class_num=10,batch_size=128,iters=120000,learning_rate=0.0001,keep_prob=1,param=param)
    Runer.TrainNet()

def load_mnist(image_dir, split='train'):
    print ('Loading MNIST dataset.')

    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(image_dir, image_file)
    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f)
    images = mnist['X'] / 127.5 - 1
    labels = mnist['y']
    labels=np.squeeze(labels).astype(int)

    return images,labels
def load_svhn(image_dir, split='train'):
    print ('Loading SVHN dataset.')

    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join(image_dir, image_file)
    svhn = scipy.io.loadmat(image_dir)
    images1 = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
    images=np.array([color.rgb2gray(im) for im in images1])
    images=images.reshape(len(images),32,32,1)
    # ~ images= resize_images(images)
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels

def load_USPS(image_dir,split='train'):
    print('Loading USPS dataset.')
    image_file='USPS_train.pkl' if split=='train' else 'USPS_test.pkl'
    image_dir=os.path.join(image_dir,image_file)
    with open(image_dir, 'rb') as f:
        usps = pickle.load(f)
    images = usps['data']
    images=np.reshape(images,[-1,32,32,1])
    labels = usps['label']
    labels=np.squeeze(labels).astype(int)
    return images,labels

def load_syn(image_dir,split='train'):
    print('load syn dataset')
    image_file='synth_train_32x32.mat' if split=='train' else 'synth_test_32x32.mat'
    image_dir=os.path.join(image_dir,image_file)
    syn = scipy.io.loadmat(image_dir)
    images = np.transpose(syn['X'], [3, 0, 1, 2]) / 127.5 - 1
    labels = syn['y'].reshape(-1)
    return images,labels


def load_mnistm(image_dir,split='train'):
    print('Loading mnistm dataset.')
    image_file='mnistm_train.pkl' if split=='train' else 'mnistm_test.pkl'
    image_dir=os.path.join(image_dir,image_file)
    with open(image_dir, 'rb') as f:
        mnistm = pickle.load(f)
    images = mnistm['data']

    labels = mnistm['label']
    labels=np.squeeze(labels).astype(int)
    return images,labels

def load_fakemnist(image_dir):
    print('load synthetic mnist dataset')
    image_file = 's_trainFakemnist.mat'
    image_dir = os.path.join(image_dir, image_file)
    s = scipy.io.loadmat(image_dir)

    images1 = s['x'] / 127.5 - 1
    images=np.array([color.rgb2gray(im) for im in images1])
    images=images.reshape(len(images),32,32,1)
    labels = s['y'].reshape(-1)
    return images,labels


def load_realsvhn(image_dir):
    print('load realsvhn dataset')
    image_file = 's_trainRealSVHN.mat'
    image_dir = os.path.join(image_dir, image_file)
    s = scipy.io.loadmat(image_dir)

    images1 = s['x'] / 127.5 - 1
    images=np.array([color.rgb2gray(im) for im in images1])
    images=images.reshape(len(images),32,32,1)
    labels = s['y'].reshape(-1)
    return images,labels
    print("success")

def load_realmnist(image_dir):
    print('load realmnist dataset')
    image_file = 's_Realmnist.mat'
    image_dir = os.path.join(image_dir, image_file)
    s = scipy.io.loadmat(image_dir)

    images1 = s['x'] / 127.5 - 1
    images=np.array([color.rgb2gray(im) for im in images1])
    images=images.reshape(len(images),32,32,1)
    labels = s['y'].reshape(-1)
    return images,labels
    print("success")

def load_testrealmnist(image_dir):
    print('load testrealmnist dataset')
    image_file = 's_testRealmnist.mat'
    image_dir = os.path.join(image_dir, image_file)
    s = scipy.io.loadmat(image_dir)

    images = s['x'] / 127.5 - 1
    labels = s['y'].reshape(-1)
    return images,labels
    print("success")
def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

if __name__=="__main__":
    main()
