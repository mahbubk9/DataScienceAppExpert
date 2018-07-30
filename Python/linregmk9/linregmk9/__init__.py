
def norm_eqn_fit(x,y):
     import numpy as np
     import matplotlib.pyplot as plt
     N=y.size
     D=len(x.shape)
     #SIMPLE LINEAR REGRESSION
     if D==1:
         dummy=np.array([[1]*N]).T
         X = np.column_stack((dummy,x))
         w=np.linalg.solve(X.T.dot(X),X.T.dot(y))
         yhat=X.dot(w)
         R2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
         fig=plt.figure()
         plt.scatter(x,y,color = "green")
         plt.plot(x,yhat)
         weights="Weights are: "
         Rsquared="Rsquared is: "
         result=( [weights, w],[Rsquared, R2],fig)
         
         return result
        
         
     #MULTIPLE LINEAR REGRESSION
     else:
         dummy=np.array([[1]*N]).T
         X = np.column_stack((dummy,x))
         w=np.linalg.solve(X.T.dot(X),X.T.dot(y))
         yhat=X.dot(w)
         R2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
         weights="Weights are: "
         Rsquared="Rsquared is: "
         result=( [weights, w],[Rsquared, R2])
         
         return result
     
              
             
         
              
     
     
     
     
     
     
     

def grad_descnt_fit(x,y,eta=0.01,epoch=2000,lmda_l2=0,lmda_l1=0):
    
    import numpy as np
    import matplotlib.pyplot as plt
    N=y.size
    D=x.shape[1]
    #SIMPLE LINEAR REGRESSION
    if D==1:
         dummy = np.array([[1]*N]).T
         X  = np.column_stack((dummy,x))
         #w = np.random.randn(D+1)
         w = np.array([np.random.randn(D+1)]).T
         J = []
         for t in range(epoch):
              yhat = X.dot(w)
   
              J.append(np.dot((y - yhat).T,y - yhat))
              w-=eta * X.T.dot(yhat-y)
         yhat = X.dot(w)
         
         R2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
         fig=plt.figure()
         plt.scatter(x,y,color = "green")
         plt.plot(x,yhat)
         weights="Weights are: "
         Rsquared="Rsquared is: "
         result=( [weights, w],[Rsquared, R2],fig)
         
         return result
     
    else:
        #MULTIPLE LINEAR REGRESSION
          dummy = np.array([[1]*N]).T
          X  = np.column_stack((dummy,x))
          w = np.random.randn(D+1)
          J = []
          for t in range(epoch):
               yhat = X.dot(w)
               J.append(np.dot((y - yhat).T,y - yhat))
               w-=eta * X.T.dot(yhat-y)
          yhat = X.dot(w)
          R2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
          weights="Weights are: "
          Rsquared="Rsquared is: "
          fig=plt.figure()
          plt.plot(J)
          result=( [weights, w],[Rsquared, R2],fig)
         
          return result
     
         
    
    
    
def predict(fit,x,y):
    import numpy as np
    import matplotlib.pyplot as plt
    #SIMPLE LINEAR PREDICTION
    D=len(fit[0][1])
    if D==2:
        yhat=fit[0][1][0]+x*fit[0][1][1]
        R2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
        fig=plt.figure()
        plt.scatter(x,y,color = "green")
        plt.plot(x,yhat)
        Rsquared="Rsquared is: "
        pred="Predictions are: "
        act="Actual Values are: "
       
        result=([Rsquared, R2],[act,y],[pred,yhat],fig)
        return result 
    #MULTIPLE LINEAR PREDICTION
    else:
        N=len(y)
        w=fit[0][1].T
        dummy = np.array([[1]*N]).T
        X  = np.column_stack((dummy,x))
        yhat=X.dot(w)
        R2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
        Rsquared="Rsquared is: "
        relerror =np.abs((y - yhat) / y)
        mre=np.median(relerror)
        MRE= "Median Relative Error: "
        act="Actual Values are: "
        pred="Predictions are: "
        result=( [MRE, mre],[Rsquared, R2],[act,y],[pred,yhat])
        
        return result
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   