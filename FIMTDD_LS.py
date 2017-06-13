__author__ = 'matsrichter'


import numpy as np

class FIMTDD:
    """
    The Learning Algorithm as Object
    """

    def __init__(self,gamma=0.01,n_min = 96,alpha=0.05,threshold=50,learn=0.01):
        """

        :param gamma:       used for hoefding-bound
        :param n_min:       time intervall for checking splits and alt-trees
        :param alpha:       used for change detection
        :param threshold:   threshold for change detection
        :return:
        """
        self.root = LeafNode(self,n_min=n_min,gamma=gamma,alpha=alpha,threshold=threshold,learn=learn)
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        self.n_min = n_min
        self.detection = True
        self.l = learn
        self.c = 0
        pass

    def eval(self,x):
        try:
            k = x[0]
        except:
            x = [x]
        #self.c += 1
        yp = self.root.eval(np.array(x))
       # print yp
        return yp

    def eval_and_learn(self,x,y):
        try:
            k = x[0]
        except:
            x = [x]
        self.c += 1
        self.root.isroot = True
        yp = self.root.eval_and_learn(np.array(x),y)
        #print str(self.c)+" ( yp: "+str(yp)+", y: "+str(y)+")"+" loss: "+str(np.fabs(yp-y))
        return yp

    def count_nodes(self):
        node = self.root
        def c_l(node):
            if type(node) == LeafNode:
                return 1
            else:
                return 1 + c_l(node.left) + c_l(node.right)
        sol = c_l(node)
        return sol

    def count_leaves(self):
        node = self.root
        def c_l(node):
            if type(node) == LeafNode:
                return 1
            else:
                return c_l(node.left) + c_l(node.right)
        sol =  c_l(node)
        return sol

class Node:

    i_c = 0

    def __init__(self,parent,key=None,key_dim=0,left=None,right=None,alpha=0.005,threshold=50,n_min=100,gamma=0.01,learn=0.1):
        """

        :param parent:      The parent of the node
        :param key:         the key of the node
        :param key_dim:     indicates the attribute of the key value
        :param left:        left child-node
        :param right:       right child-node
        :param alpha:       alpha value for change-detection
        :param threshold:   threshold for change detection
        :param n_min:       minimum time period for split and subtree replacement
        :param gamma:       value for the hoefding-bound
        :param learn:       learning rate of the leafnote
        :return:
        """
        self.gamma = gamma
        self.c_x = 0
        self.y = 0.0
        self.y_sq = 0.0
        self.alpha = alpha
        self.n_min = n_min
        self.key = key
        self.key_dim = key_dim
        self.parent = parent
        self.left = left
        self.right = right
        self.isLeaf = False
        Node.i_c += 1
        self.index = Node.i_c #self.parent.index +1
        self.cumloss = 0.0
        self.minPH = None
        self.PH = 0.0
        self.threshold = threshold
        #turns on or of change detection
        self.detection = True
        self.alt_tree = None
        self.isAlt = False
        self.sq_loss = 0.0
        self.cum_sq_loss = 0.0
        self.alt_counter = 0
        self.isroot = False
        self.update_root()
        self.l = learn
        self.S_i = 0

    def update_root(self):
        if isinstance(self.parent,FIMTDD):
            self.isroot = True
        else:
            self.isroot = False

    def update_alt(self,val):
        self.isAlt = val
        if self.left != None:
            self.left.update_alt(val)
        if self.right != None:
            self.right.update_alt(val)

    def eval(self,x):
        """

        :param x:   data point
        :return:    prediction
        """
        if x[self.key_dim] <= self.key:
            yp = self.left.eval(x)
        else:
            yp = self.right.eval(x)
        return yp

    def eval_and_learn(self,x,y):
        """

        :param x:   data point
        :param y:   label
        :return:    prediction
        """
        #increment seen sample counter
        self.c_x += 1
        #init prediction-variable
        yp = 0.0
        #get prediction
        if x[self.key_dim] <= self.key:
            yp = self.left.eval_and_learn(x,y)
        else:
            yp = self.right.eval_and_learn(x,y)

        #change detection segment start
        if self.alt_tree != None:
            #increment alt_counter if alternate tree exists
            self.alt_counter += 1
            self.alt_tree.eval_and_learn(x,y)
        #update squared error
        self.sq_loss = (y - yp)**2
        self.S_i = (self.S_i*0.995) + self.sq_loss
        if self.alt_tree != None and self.alt_counter%self.n_min == 0 and self.alt_counter != 0:
            #check all n_min samples the q statistics of current and alt-tree
            if self.c_x == 0:
                this_q = 0.0
                alt_q = 1.0
            else:
                this_q = self.S_i
                alt_q = self.alt_tree.S_i
            if alt_q == 0:
                alt_q = 0.00000001
            if not this_q == 0.0 and np.log(this_q/alt_q) > 0:
                #if alt-tree has better performance, replace this node with alternate subtree
                #print "Replaced alt tree (LS) @",self.index
                if self.isroot:
                    self.parent.root = self.alt_tree
                elif self.parent.left.index == self.index:
                    self.parent.left = self.alt_tree
                else:
                    self.parent.right = self.alt_tree
                self.alt_tree.parent = self.parent
                self.alt_tree.update_alt(False)
                self.alt_tree.detection = True
            if self.alt_counter >= self.n_min*10:
                #if alternate tree is still not better than the current one, remove it
                self.alt_tree = None
                self.alt_counter = 0
        self.cum_sq_loss += self.sq_loss
        #change detection segment end

        if self.detect_change(y,yp) and self.detection and not self.isAlt and self.alt_tree is None:
            #avoid change detection on higher levels and grow subtree
            self.parent.detection = False
            self.grow_alt_tree()
        #elif not self.detection and not self.isAlt:
        #    #activate change detection if this node is not root of a subtree
        #    self.parent.detection = False
        #    self.detection = True
        if self.alt_tree != None or (not self.detection and not self.isAlt):
            #deactivate change detection on all higher level nodes if low level change detection is allready triggered
            self.parent.detection = False
            self.detection = False
        else:
            self.detection = True
        return yp


    def grow_alt_tree(self):
        """
        creates an alternative tree
        :return:
        """
        #print "gorow alt node: "+str(self.index)
        #self.alt_tree = LeafNode(self,self.n_min,None,self.gamma,self.alpha,threshold=self.threshold,learn=self.l)
        self.alt_tree = LeafNode(self,n_min=self.n_min,model=None,gamma=self.gamma,alpha=self.alpha,threshold=self.threshold,learn=self.l)
        #self.alt_tree.index += 3

        self.alt_tree.isAlt = True
        return

    def detect_change(self,y,yp):
        """
        Page-Hinckley-Test for change detection

        :param y:   the true label value
        :param yp:  the prediction
        :return:    true if change is detected, else false
        """
        #return False

        error = np.fabs(y-yp)
        self.cumloss += error

        self.PH += error - (self.cumloss/self.c_x) - self.alpha

        if self.minPH is None or self.PH < self.minPH :
            self.minPH = self.PH
        return self.PH - self.minPH > self.threshold
        #return False

class LeafNode(Node):
    """
    LeafNode-Object for FIMTDD
    """
    def __init__(self,parent,n_min,model=None,gamma=0.01,alpha=0.005,threshold=50,learn=0.1):
        """

        :param parent:      parent node
        :param n_min:       minimum intervall for split and alt-tree replacement
        :param model:       the perceptron
        :param gamma:       hoefding-bound value
        :param alpha:       used for change detection
        :param threshold:   threshold for change detection
        :return:
        """
        Node.__init__(self,parent,None,0,None,None,alpha=alpha,threshold=threshold,n_min=n_min)
        self.isLeaf = True
        self.n_min = n_min
        self.gamma = gamma
        self.alpha = alpha
        self.l = learn
        if model is None:
            self.model = LinearRegressor(self)
        else:
            self.model = model
        #EBST-Tree for storing data,used for splitting
        self.ebst = None
        self.c = 0
        pass

    def split(self,splits,index):
        """

        :param splits: dictionary containing the best split
        :param index:  index of the best split, indicating the attribute (dimension in data-vector)
        :return:
        """
        #return
        #print "splitting node at index: "+str(index)
        node = Node(self.parent,n_min=self.n_min,key_dim=index,key=splits['bestsplit'],gamma=self.gamma,learn = self.l,threshold=self.threshold,alpha=self.alpha)
        left = LeafNode(parent=node,n_min=self.n_min,gamma=self.gamma,alpha=self.alpha,learn = self.l,threshold=self.threshold)
        right = LeafNode(parent=node,n_min=self.n_min,gamma=self.gamma,alpha=self.alpha,learn = self.l,threshold=self.threshold)
        l1 = LinearRegressor(left,self.model.filter.w,learn = self.l)
        l2 = LinearRegressor(right,self.model.filter.w,learn = self.l)
        left.model = l1
        right.model = l2
        #left.index += 1
        #right.index += 2
        node.left = left
        node.right = right
        try:
            if self.isroot:
                self.parent.root = node
                node.update_root()
            elif self.parent.left.index == self.index:
                self.parent.left = node
            elif self.parent.right.index == self.index:
                self.parent.right = node
            else:
                self.parent.alt_tree = node
                node.update_alt(True)
        except:
            self.parent.root = node
            node.update_root()

    def eval(self,x):
        """

        :param x:   data point
        :return:    prediction
        """
        return self.model.eval(x)

    def eval_and_learn(self,x,y):
        """

        :param x:   data point
        :param y:   label
        :return:    prediction
        """
        #increment counters (currently unused)
        self.c += 1
        self.c_x += 1
        self.y += y
        self.y_sq += y**2
        #get prediction from perceptron
        yp = self.model.eval_and_learn(x,y)

        #change detection segment start (exact same same as in Node)
        if self.alt_tree != None:
            self.alt_counter += 1
            self.alt_tree.eval_and_learn(x,y)
        self.sq_loss = (y - yp)**2
        self.S_i = (self.S_i*0.995) + self.sq_loss
        if self.alt_tree != None and self.alt_counter%self.n_min == 0 and self.alt_counter != 0:
            if self.alt_tree.c_x == 0:
                this_q = 0.0
                alt_q = 0.0
            else:
                this_q = self.S_i
                alt_q = self.alt_tree.S_i
            if not this_q == 0.0 and np.log(this_q/alt_q) > 0:
                self.update_root()
                if self.isroot:
                    self.parent.root = self.alt_tree
                elif self.parent.left.index == self.index:
                    self.parent.left = self.alt_tree
                else:
                    self.parent.right = self.alt_tree
                self.alt_tree.isAlt = False
                self.alt_tree.detection = True
                self.alt_tree.parent = self.parent
            if self.alt_counter >= self.n_min*10:
                self.alt_tree = None
                self.alt_counter = 0
        self.cum_sq_loss += self.sq_loss
        #change detection segment end

        if self.detect_change(y,yp) and self.detection and not self.isAlt and self.alt_tree is None:
            self.parent.detection = False
            self.grow_alt_tree()
        elif not self.detection and not self.isAlt:
            self.parent.detection = False
            self.detection = True
        if self.alt_tree != None or (not self.detection and not self.isAlt):
            #deactivate change detection on all higher level nodes if low level change detection is allready triggered
            self.parent.detection = False
            self.detection = False
        else:
            self.detection = True
        if self.ebst is None:
            self.ebst = list()
            try:
                for xi in x:
                    tree = E_BST()
                    self.ebst.append(tree)
            except:
                tree = E_BST()
                self.ebst.append(tree)
        for i in range(len(self.ebst)):
            self.ebst[i].add(x[i],y)
        if self.c == self.n_min:
            #try to split
            self.c = 0
            splits = list()
            for tree in self.ebst:
                #find best splits
                splits.append(self.findBestSplit(tree))
            bi = int(self.findBest(splits))
            bound = 1-self.hoefding_bound(splits[bi]['n'])
            if splits[bi]['score'] < bound or self.hoefding_bound(splits[bi]['n']) < 0.05 or len(splits) == 1:
                self.split(splits[bi],bi)
        return yp


    def findBest(self,splits):
        """

        :param splits:  list of dictionaries containing the best split per attribute
        :return:        index of the dictionary with the best split over all attributes
        """
        max_index = None
        second_place = None
        for i in range(len(splits)):
            m = splits[i]['max']
            if max_index is None or m > max_index:
                second_place = max_index
                max_index = i

        if second_place != None:
            splits[max_index]['score'] = splits[second_place]['max']/splits[max_index]['max']
        return max_index


    def findBestSplit(self,tree,sdr = None):
        """
        Recursively calculate the best split of an attribute
        :param tree:    A EBST-tree
        :param sdr:     dictionary with global information for the tree search
        :return:        dictionary with best split and some additional information
        """
        assert(isinstance(tree,E_BST))
        if sdr is None:
            sdr = dict()
            sdr['sumtotalLeft'] = 0.0
            sdr['sumtotalRight'] = tree.root.l_y + tree.root.r_y
            sdr['sumsqtotalLeft'] = 0.0
            sdr['sumsqtotalRight'] = tree.root.l_y_sq + tree.root.r_y_sq
            sdr['righttotal'] = tree.root.l_count + tree.root.r_count
            sdr['total'] = sdr['righttotal']
            sdr['n'] = sdr['total']
            sdr['max'] = None

        if tree.root.left != None:
            self.findBestSplit(E_BST(tree.root.left),sdr)
        sdr['sumtotalLeft'] = sdr['sumtotalLeft'] + tree.root.l_y
        sdr['sumtotalRight'] = sdr['sumtotalRight'] - tree.root.l_y
        sdr['sumsqtotalLeft'] = sdr['sumsqtotalLeft'] + tree.root.l_y_sq
        sdr['sumsqtotalRight'] = sdr['sumsqtotalRight'] - tree.root.l_y_sq
        sdr['righttotal'] = sdr['righttotal'] - tree.root.l_count

        new_sdr = self.computeSDR(sdr)
        if(sdr['max'] is None or new_sdr > sdr['max']):
            sdr['2nd'] = sdr['max']
            sdr['max'] = new_sdr
            try:
                if not new_sdr == 0.0:
                    sdr['score'] = new_sdr#sdr['2nd'] / new_sdr
                else:
                    sdr['score'] = 1.0
            except:
                sdr['score'] = 1.0
            sdr['bestsplit'] = tree.root.key

        if tree.root.right != None:
            self.findBestSplit(E_BST(tree.root.right),sdr)
        sdr['sumtotalLeft'] = sdr['sumtotalLeft'] - tree.root.l_y
        sdr['sumtotalRight'] = sdr['sumtotalRight'] + tree.root.l_y
        sdr['sumsqtotalLeft'] = sdr['sumsqtotalLeft'] - tree.root.l_y_sq
        sdr['sumsqtotalRight'] = sdr['sumsqtotalRight'] + tree.root.l_y_sq
        sdr['righttotal'] = sdr['righttotal'] + tree.root.l_count
        #sdr['total'] = sdr['righttotal']
        return sdr

    def sd(self,n,y_sq_count, y_count):
        if n == 0:
            return 0.0
        n_inv = 1/float(n)
        return np.sqrt(np.fabs(n_inv*(y_sq_count - (n_inv*(y_count**2)))))

    def computeSDR(self,sdr):
        """
        calculate the standard devitation reduction
        :param sdr:     dictionary from the findBestSplit-Function
        :return:        SDR-value (float)
        """
        n_l = sdr['total']- sdr['righttotal']
        n_r = sdr['righttotal']
        l_s = sdr['sumtotalLeft']
        l_s_sq = sdr['sumsqtotalLeft']
        r_s = sdr['sumtotalRight']
        r_s_sq = sdr['sumsqtotalRight']
        total = float(n_l+n_r)
        base = self.sd(n_l+n_r, l_s_sq+r_s_sq, l_s+r_s)
        sd_l = self.sd(n_l,l_s_sq,l_s)
        ratio_l = n_l/total
        sd_r = self.sd(n_r,r_s_sq,r_s)
        ratio_r = n_r/total
        return base - (ratio_l*sd_l) - (ratio_r*sd_r)

    def hoefding_bound(self,n):
        """

        :param n:   the totalnumber of samples seen
        :return:    hoefding-bound
        """
        log = np.log(1.0/self.gamma)
        n = 2*n
        result = np.sqrt(log/n)
        #result =  np.sqrt(np.log(1.0/self.gamma)/(2.0*n))
        return result

import padasip as pa
class LinearRegressor:
    """
    A perceptron for FIMTDD
    """

    def __init__(self,leafnode,w=None,learn = 0.01):
        self.leafnode = leafnode
        self.l = learn
        self.covM = 10 ** 3
        if w is None:
            self.w = w
        else:
            self.w = np.random.rand(len(w))
            self.w = self.w / np.linalg.norm(self.w)
            k = list()
            for i in range(len(self.w)):
                self.w[i] = w[i]
                k.append(w[i])
            self.S = self.covM * np.identity(len(self.w))
            self.filter = pa.filters.FilterRLS(len(self.w))
            self.filter.w = k
        self.x_count = None
        self.x_sq_count = None
        self.c = 0.0

        self.forgF = 1.0

    def eval(self,x):

        if self.x_count is None:
            self.x_count = np.zeros(len(x))
            self.x_sq_count = np.zeros(len(x))
        #x = self.normalize(x,0)
        x = np.hstack((1.0,x))
        if self.w is None:
            self.w = np.random.rand(len(x))
            self.w = self.w/np.linalg.norm(self.w)
            self.S = self.covM * np.identity(len(self.w))
            self.filter = pa.filters.FilterRLS(len(self.w))
        #yp = np.inner(x,self.w)
        yp = self.filter.predict(x)
        return yp

    def eval_and_learn(self,x,y):
        yp = self.eval(x)
        self.x_count += x
        self.x_sq_count += x**2
        self.c += 1.0
        #x = self.normalize(x,y)
        x = np.hstack((1.0,x))
        self.learn(x,y,yp)
        return yp

    def rls_learn(self, x, phiX, y, yp):
        deltaAlpha = np.dot(self.S, phiX) / ((self.forgF + np.inner(phiX, np.dot(self.S, phiX))) * (y - yp))

        #print deltaAlpha

        self.S = self.S / self.forgF - np.outer(np.dot(self.S, phiX), np.dot(phiX, self.S)) \
                                       / (self.forgF * (self.forgF + np.inner(np.dot(phiX, self.S), phiX)))

        #print self.S
        self.filter.adapt(y,x)
        return deltaAlpha

    def learn(self,x,y,yp):
        delta = self.l * (y - yp)*x
        self.rls_learn(x,self.w,y,yp)
        self.w += delta
        #self.w = self.w/np.linalg.norm(self.w)

    def normalize(self,x,y):
        sd = self.leafnode.sd(self.c,self.x_sq_count,self.x_count)
        avg = self.x_count/self.c
        if sd == 0:
            return x
        norm = (x - avg)/(3*sd)
        return norm

    def denormalize(self,x,y):
        pass

class E_BST:

    def __init__(self,root = None):
        self.root = root

    def add(self,key,y):
        if self.root is None:
            self.root = Node_EBST(key,y)
        else:
            self.root.add(key,y)

class Node_EBST:

    def __init__(self,x,y,parent = None):
        self.key = x
        self.parent = parent
        self.left = None
        self.right = None

        self.l_count = 1
        self.l_y = y
        self.l_y_sq = y**2

        self.r_count = 0
        self.r_y = 0
        self.r_y_sq = 0

    def add(self,val,y):
        if val <= self.key:
            self.l_count += 1
            self.l_y += y
            self.l_y_sq += y**2
            if self.left is None and val != self.key:
                self.left = Node_EBST(val,y,self)
            elif val == self.key:
                pass
            else:
                self.left.add(val,y)
        else:
            self.r_count += 1
            self.r_y += y
            self.r_y_sq += y**2
            if self.right is None:
                self.right = Node_EBST(val,y,self)
            else:
                self.right.add(val,y)
        return