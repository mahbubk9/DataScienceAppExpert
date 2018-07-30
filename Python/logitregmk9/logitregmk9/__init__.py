import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def sigmoid(h):
    return 1/(1+np.exp(-h))

def binary_cross_entropy(y,p):
    return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))

def binary_classification_rate(y,p):
    return np.mean(np.round(p)==y)

def softmax(H):
    eH=np.exp(H)
    return eH/eH.sum(axis=1,keepdims=True)

def multi_classification_rate(y,P):
    return np.mean(y==P.argmax(axis=1))

def multi_cross_entropy(Y,P):
    return -np.sum(Y*np.log(P))



#LOGISTIC FIT FUNCTION
def fit(X,y,eta=0.001,epoch=2000,lmda_l2=0,lmda_l1=0):
    
    K=len(set(y))
    N=len(y)
    D=X.shape[1]
    #BINARY CLASSIFICATION
    if K==2:
        w=np.random.randn(D+1)
        PHI=np.column_stack((np.array([[1]*N]).T,X))
        J=[]
        
        for i in range (epoch):
            p=sigmoid(PHI.dot(w))
            J.append(binary_cross_entropy(y,p))
            w-=eta*PHI.T.dot(p-y)

    
        p=sigmoid(PHI.dot(w)) 
        jplot=plt.figure()
        plt.title('Cross Entropy Plot for Logistic Regression')
        plt.plot(J)
        cls_rate=binary_classification_rate(y,p)
        clrt="Classification Rate is: "
        roc_matrix=np.column_stack((p,np.round(p),y))
        roc_matrix=roc_matrix[roc_matrix[:,0].argsort()[::-1]]
        tp=np.cumsum((roc_matrix[:,1]==1)& (roc_matrix[:,2]==1))/np.sum(roc_matrix[:,2]==1)
        fp=np.cumsum((roc_matrix[:,1]==1) & (roc_matrix[:,2]==0))/np.sum(roc_matrix[:,2]==0)
        tp=np.array([0]+tp.tolist()+[1])
        fp=np.array([0]+fp.tolist()+[1])
        roc_curve=plt.figure()
        plt.step(fp,tp)
        auc="AUC is: "
        jhead="Cost Funciton"
        wts="Weights are:"
        roc="ROC Curve"
        AUC=np.sum(np.array([u*v for u,v in zip(tp[1:],[j-i for i,j in zip(fp,fp[1:])])]))
        result=([wts,w], [clrt, cls_rate],[auc, AUC],[jhead,jplot],[roc,roc_curve])
         
        return result
    else:                # MULTINOMIAL CLASSIFICATION
        Y=np.zeros((N,K))
        for i in range(N):
            Y[i,y[i]]=1
        PHI=np.column_stack((np.array([[1]*N]).T,X))
        W=np.random.randn(D+1,K)
        J=[]

        for i in range(epoch):
           P=softmax(PHI.dot(W))
           J.append(multi_cross_entropy(Y,P))
           W-=eta*PHI.T.dot(P-Y)

        P=softmax(PHI.dot(W))
        jplot=plt.figure()
        plt.title('Cross Entropy Plot for Logistic Regression')
        jplot=plt.plot(J)
        wts="Weights are:"
        
        cls_rate=multi_classification_rate(y,P)
        clrt="Classification Rate is: "
        
        result=( [wts,W],[clrt, cls_rate])
        
        return result
    
   


    
        
def predict(fit,X,y):
         K=len(set(y))
         N=len(y)
        # D=X.shape[1]
         w=fit[0][1]
         if K==2:
             PHI=np.column_stack((np.array([[1]*N]).T,X))
             p=sigmoid(PHI.dot(w))
             cls_rate=binary_classification_rate(y,p)
             clrt="Classification Rate is: "
             roc_matrix=np.column_stack((p,np.round(p),y))
             roc_matrix=roc_matrix[roc_matrix[:,0].argsort()[::-1]]
             tp=np.cumsum((roc_matrix[:,1]==1)& (roc_matrix[:,2]==1))/np.sum(roc_matrix[:,2]==1)
             fp=np.cumsum((roc_matrix[:,1]==1) & (roc_matrix[:,2]==0))/np.sum(roc_matrix[:,2]==0)
             tp=np.array([0]+tp.tolist()+[1])
             fp=np.array([0]+fp.tolist()+[1])
             roc_curve=plt.figure()
             plt.step(fp,tp)
             auc="AUC is: "
             #jhead="Cost Funciton"
             wts="Weights are:"
             roc="ROC Curve"
             AUC=np.sum(np.array([u*v for u,v in zip(tp[1:],[j-i for i,j in zip(fp,fp[1:])])]))
             result=([wts,w], [clrt, cls_rate],[auc, AUC],[roc,roc_curve])
             
             
             
             return result
         
            
            
            
            
         else:
              W=fit[0][1]
              PHI=np.column_stack((np.array([[1]*N]).T,X))
              Phat=softmax(PHI.dot(W))
              
              cls_rate=multi_classification_rate(y,Phat)
              clrt="Classification Rate is: "
              cm=pd.crosstab(y, np.round(Phat.argmax(axis=1)), rownames=['Actual'], colnames=['Predicted'], margins=True)
              confmat="The Confusion Matrix"
              result=( [confmat,cm],[clrt, cls_rate])
        
              return result
    
         
            
             
             
             
             
             
             
             
             
             
            