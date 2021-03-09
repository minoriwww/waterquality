# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import ShuffleSplit, KFold


__all__ = ['BinaryClassficationBootstraping', 'MultiClassficationBootstraping']

class BinaryClassficationBootstraping(ShuffleSplit):
    """Bootstraping, a class inheirt from cross-validator of Sklearn
    use it like KFold or ShuffleSplit

    this class is for binary classfication
    binary classfication tested. (y=(-1,1))

    worked when data is imbalance between position and negative

    tested in py2.7
    dltdc
    """
    def __init__(self, x=None, y=None, xy=None,
        y_col_num=1, n_splits=5, test_size=0.1, train_size=None, random_state=96):

        super(BinaryClassficationBootstraping, self).__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)

        self.random_state = random_state
        np.random.seed(self.random_state)
        self.x = x #ndarray
        self.y = y #ndarray
        self.xy = xy #ndarray
        self.y_col_num = y_col_num

        self.slength = None
        self.nclass = None
        self.train_pos_s = None
        self.train_neg_s = None
        self.data_all = None
        # if self.xy is None:
        #     self.x_ = self.x.reshape((self.y.shape[0], -1))
        #     self.xy = np.concatenate((y, x_), axis=-1)
        # if self.x is None and self.y is None:
        #     self.y, self.x = self.xy[:, 0:self.y_col_num], self.xy[:, self.y_col_num:]



    def cvt_xy__y_x(self):
        self.y, self.x = self.xy[:, 0:self.y_col_num], self.xy[:, self.y_col_num:]
        return self.y, self.x

    def cvt_x_y__xy(self):
        self.x_ = self.x.reshape((self.y.shape[0], -1))
        self.xy = np.concatenate((y, x_), axis=-1)
        return self.xy

    def get_pos_neg_x_y(self):
        """get pos and neg data for binary classfication (onehot)

        Returns:
            TYPE: Description
        """
        data_pos_s, data_neg_s, data_all = self.get_pos_neg_xy(self.xy, positive_col=0)
        data_pos_y = data_pos_s.ix[:, 0:y.shape[1]].copy()
        data_pos_x = data_pos_s.ix[:, y.shape[1]:].copy()
        data_neg_y = data_neg_s.ix[:, 0:y.shape[1]].copy()
        data_neg_x = data_neg_s.ix[:, y.shape[1]:].copy()
        return data_pos_y, data_pos_x, data_neg_y, data_neg_x, data_all

    def get_pos_neg_xy(self, xy, positive_col=0,
        positive_num=1, negative_num=0,
        is_posneg_balance=False,
        #  y_col_arr=y.shape[1:]
        ):
        """
        Args:
            positive_col (int, optional): # for one hot
            positive_num (int, optional): # for other numbers (no onehot), like 1234
            negative_num (int, optional): # for other numbers (no onehot), like 1234

        Returns:
            data_pos_s, data_neg_s, data_all: Dataframe
        """
        pos_indices_arr, neg_indices_arr = self.get_pos_neg_indices(positive_col=0, positive_num=1, negative_num=0)
        data_pos = xy[pos_indices_arr]
        data_neg = xy[neg_indices_arr]
        data_pos = pd.DataFrame(data_pos)
        data_neg = pd.DataFrame(data_neg)

        data_pos_s = data_pos.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        if is_posneg_balance:
            data_neg_s = data_neg.sample(data_pos.shape[0], random_state=self.random_state).reset_index(drop=True)
        else:
            data_neg_s = data_neg.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        data_all = pd.concat([data_pos_s, data_neg_s])
        return data_pos_s, data_neg_s, data_all

    def get_val_xy(self, val_xy=None, train_pos=None, train_neg=None):
        """Summary
        Args:
            val_xy (ndarray): Description
        Returns:
            TYPE: Description
        """
        if(val_xy is not None): # use all data as val
            _, _, val_all = self.get_pos_neg_xy(val_xy, is_posneg_balance=False)
            #(355340,1,33,21) after extract same size as positive (48050,1,33,21)
        else:
            a = int(train_pos.shape[0]*(1 - self.test_size))
            b = train_neg.shape[0]-int(train_pos.shape[0]*self.test_size)
            self.train_pos_s = train_pos[0:a]
            self.train_neg_s = train_neg[0:b]

            val_pos = train_pos[(a+1):]
            val_neg = train_neg[b+1:]

            val_all = pd.concat([val_pos,val_neg])
            self.slength, self.nclass = self.calculate_nclass(self.train_pos_s.shape[0], self.train_neg_s.shape[0])
        return val_all

    ########################## data processing ######################################

    def gen_fold_alldata(self, pos_arr, neg_arr, slength, nclass, pos_rate=1, neg_rate=1):
        """generator for all data, combine pos and neg
        Args:
            pos_arr (ndarray): Description
            neg_arr (ndarray): Description
            slength (int): Description
            nclass (int): Description
            pos_rate=1, neg_rate=1: less than 1, ratio of positive and negative
        Yields:
            ndarray: all data including pos and neg, shuffled
        """
        np.random.seed(self.random_state)
        np.random.shuffle(neg_arr)
        pos_arr = pos_arr[np.random.choice(pos_arr.shape[0], int(slength*pos_rate), replace=False )]

        for t in range(nclass):
            train_neg_ss = neg_arr[( int(slength*neg_rate) * t):(int(slength*neg_rate) * t+int(slength*neg_rate) ) ]
            data_all = np.concatenate((pos_arr, train_neg_ss), axis=0)
            np.random.shuffle(data_all)
            yield data_all

    def bootstraping(self,
        train_xy=None,
        val_xy=None,
        bootstrap_num=1
        ):
        """
        Para
        ---
        bootstrap_num: outside loop

        Yield
        ---
        train_all:ndarray, test_all:ndarray
        train and test of each fold
        """
        self.train_pos_s, self.train_neg_s, self.data_all = self.get_pos_neg_xy(train_xy)

        self.slength, self.nclass = self.calculate_nclass(self.train_pos_s.shape[0], self.train_neg_s.shape[0])

        val_all = self.get_val_xy(train_pos=self.train_pos_s, train_neg=self.train_neg_s)

        maxneg = self.n_splits
        if(maxneg is not None):
            self.nclass=min(maxneg, self.nclass) #cannot do more than maxneg times

        for bootstrap_num_i in range(bootstrap_num):
            for data_all in self.gen_fold_alldata(self.train_pos_s.values, self.train_neg_s.values, self.slength, self.nclass):
                yield data_all, val_all.as_matrix()


    ######################### tools ###############################

    def calculate_nclass(self, n_pos, n_neg, srate=0.8):
        """calculate how many times the pos+neg operation will execute
        Args:
            srate (float, optional): ratio to keep the positive sample
        Returns:
            sampleweights: ndarray
        """
        self.slength = int(n_pos * srate); #transfer 0.1 to val so update self.slength
        self.nclass = int(n_neg / self.slength)
        return self.slength, self.nclass

    def assign_sample_weight(self, train_all, hw_res=None):
        """Summary

        Args:
            train_all (TYPE): Description
            hw_res (None, optional): Description

        Returns:
            ndarray: sampleweights
        """
        sampleweights = None
        if (hw_res is not None):
            sampleweights = np.ones(len(train_all))
            sampleweights[np.where(train_all.as_matrix()[:,0] == hw_res)] *= sum(sampleweights[np.where(train_all.as_matrix()[:,0] != 0)])/sum(sampleweights[np.where(train_all.as_matrix()[:,0]==hw_res)])
        return sampleweights

    def assign_class_weight(self, train_all, hc_res, hc_res2):
        """Summary

        Args:
            train_all (TYPE): Description
            hc_res (TYPE): Description
            hc_res2
        Returns:
            dict: classweights
        """
        classweights=None
        if(hc_res is not None):
            classweights={0:1,1:1,2:1,3:1} #0 negative, 1 S 2 T 3 Y
            classweights[hc_res]=sum(train_all.as_matrix()[:,0]!=0)/sum(train_all.as_matrix()[:,0]==hc_res)

        if(hc_res2 is not None): #negative has weight!
            classweights={0:1,1:1,2:1,3:1,4:1,5:1} #sp 0 tp 1 yp 2 sn 3 tn 4 yn 5
            classweights[hc_res2[0]]=sum(train_all.as_matrix()[:,0]<=2)/sum(train_all.as_matrix()[:,0]==hc_res2[0])
            classweights[hc_res2[1]]=sum(train_all.as_matrix()[:,0]<=2)/sum(train_all.as_matrix()[:,0]==hc_res2[1])
        return classweights

    def get_pos_neg_indices(self, positive_col=0, # for one hot
        positive_num=1, negative_num=0):
        """return positive and neg indices as 1d ndarray, (n,)
        """
        pos_indices_arr = np.array(np.where(self.xy[:, positive_col] == positive_num))
        neg_indices_arr = np.array(np.where(self.xy[:, positive_col] == negative_num))
        return pos_indices_arr[0], neg_indices_arr[0]

    ######################### inheirt ###############################



    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        inheratance overwrite, boostraping
        Parameters
        ----------
        X : nd array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        np.random.seed(self.random_state)

        pos_indices_arr, neg_indices_arr = self.get_pos_neg_indices(positive_col=0, positive_num=1, negative_num=0)

        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        slength, nclass = self.calculate_nclass(len(pos_indices_arr), len(neg_indices_arr), srate=1)
        maxneg = self.n_splits
        if(maxneg is not None):
            nclass=min(maxneg, nclass) #cannot do more than maxneg times

        for data_all in self.gen_fold_alldata(pos_indices_arr, neg_indices_arr, slength, nclass):
            split_position = int(data_all.shape[0]*(1 - self.test_size))
            train_all = data_all[:split_position]
            test_all = data_all[split_position:]
            yield train_all, test_all

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits



class MultiClassficationBootstraping(ShuffleSplit):
    """
    X: ndarray, X[0] is sample index
    y: 1d ndarray or list []
    k: number of fold
    group 1: number of samples are the same in each class
          2: number of samples might be different in each class
          3: special case for water quality project
    indexList: number of samples in each class
               default None in group 1 case
               this parameter is required in group 2 case

    example:
    ---
    kf = KFold(data,label,5,1)
    kf.save_index('vgg_index.csv')
    for i in range(0,5):
        train,label,vali_train,vali_label = kf.getItem(i)
        label=to_categorical(label)
        vali_label=to_categorical(vali_label)
    ---
    tested py2 and 3, unbalance class not been tested

    zzcz4
    """
    def __init__(self, X, y, k, group=1, indexList=None, random_state=96):
        super(MultiClassficationBootstraping, self).__init__(
            n_splits=k,
            test_size=0.1,
            random_state=random_state)
        self.X = X
        self.y = y
        self.k = k
        self.group = group
        self.indexList = indexList
        self._order = self.generateOrder()
        np.random.seed(self.random_state)

    def generateRandomInt(self,num):
        """
        num: number of samples in each class(int)
        list: number of samples for one class in each fold
        """
        if num < self.k:
            raise ValueError('KFold only holds %i folds' % num)
        val = int(num / self.k)
        list = [val]*self.k
        # check remainder
        if num-val*self.k != 0:
            for i in range(0,num-val*self.k):
                list[i] += 1
        return list


    def generateRandomList(self,num):
        """
        num: number of samples in each class(list: 1×class number)
             this function is used in group 2 case
        list: number of samples for one class in each fold
              (list: class number×fold number)
        """
        list = []
        for idx in range(0,len(num)):
            list.append(self.generateRandomInt(num[idx]))
        return list


    def generateOrder(self):
        """
        return the index of samples in each fold according to the random list generated above
        """
        # list record the value is selected in each fold
        list = np.zeros(self.X.shape[0])
        res = []
        # number of classes
        classNum = len(np.unique(self.y))
        if self.group == 2:
            return self.generate_order_unbalanced()
        num = int(self.X.shape[0]/classNum)
        ranList = self.generateRandomInt(num)
        for i in range(0,len(ranList)-1):
            index = []
            for idx in range(0,classNum):
                # not optimize
                flag = True
                resultList = []
                while flag:
                    resultList = random.sample(range(idx*num,(idx+1)*num),ranList[i])
                    flag = False
                    for j in range(len(resultList)):
                        if list[resultList[j]] != 0:
                            flag = True
                            break
                # random list get√
                index = index + resultList
            res.append(index)
            # del those selected num
            for j in range(0,len(index)):
                list[index[j]] = 1
        # the last fold
        index = []
        for j in range(0,len(list)):
            if list[j] == 0:
                index += [j]
        res.append(index)
        self._order = res
        return res

    def generate_order_unbalanced(self):############## TODO ############
        """
        return the index of samples in each fold according to the random list generated above
        """
        # list record the value is selected in each fold
        list = np.zeros(self.X.shape[0])
        res = []
        # number of classes
        classNum = len(np.unique(self.y))
        ranList = self.generateRandomList(self.indexList)
        for i in range(0,self.k-1):
            index = []
            count = 0
            for idx in range(0,len(ranList)):
                # not optimize
                flag = True
                resultList = []
                while flag:
                    resultList = random.sample(range(count,count+self.indexList[idx]),ranList[idx][i])
                    flag = False
                    for j in range(len(resultList)):
                        if list[resultList[j]] != 0:
                            flag = True
                            break
                # random list get√
                index += resultList
                count += self.indexList[idx]
            res.append(index)
            # del those selected num
            for j in range(0,len(index)):
                list[index[j]] = 1
        # the last fold
        index = []
        for j in range(0,len(list)):
            if list[j] == 0:
                index += [j]
        res.append(index)
        self._order = res
        return res


    def save_index(self, name):
        with open(name,"w") as csvfile:
            writer = csv.writer(csvfile)
            for idx in range(self.k):
                list = ["list %d" %idx] + [self._order[idx]]
                writer.writerow(list)

    def get_fold_list(self, start = 2, end = 10):
        """all sample in several class will all be deleted
        """
        res = []
        num = end - start # 6 pictures in each class expect for class 0

        result = random.sample(range(0, 100),100)
        resultList = [[result[j] for j in range(i*10,(i+1)*10)] for i in range(0,10)]
        for i in range(len(resultList)):
            list_ = []
            for j in range(len(resultList[i])):
                list_ += [resultList[i][j]*num+index for index in range(start, end)]
            res.append(list_)
        self._order = res
        return res


    def getItem(self, instance):
        """
        instance: fold number, start from 0 (int)
        return the training and testing data according to the fold index
        """
        if instance >= self.k:
            raise ValueError('KFold only holds %i folds' % self.k)

        self.n_splits = instance

        mask = np.zeros(self.X.shape[0], dtype=bool)

        for idx in range(0,len(self._order[instance])-1):
            print(instance)
            print(self._order[instance][idx])
            mask[self._order[instance][idx]] = True

        x_train = self.X[~mask]
        y_train = np.array(self.y)[~mask]
        x_test = self.X[mask]
        y_test = np.array(self.y)[mask]

        return x_train, y_train, x_test, y_test

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        inheratance overwrite, boostraping
        Parameters
        ----------
        X : nd array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        instance = self.n_splits - 1
        mask = np.zeros(X.shape[0], dtype=bool)

        for idx in range(0,len(self._order[instance])-1):
            print(instance)
            print(self._order[instance][idx]-1)
            mask[self._order[instance][idx]-1] = True
            yield np.where(mask==False), np.where(mask==True)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

if __name__ == '__main__':
    xy = pd.read_table(trainfile, sep='\t', header=None).values
