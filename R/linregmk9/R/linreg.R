# Some useful keyboard shortcuts for package authoring:
#
#   Build and Reload Package:  'Ctrl + Shift + B'
#   Check Package:             'Ctrl + Shift + E'
#   Test Package:              'Ctrl + Shift + T'


## SIMPLE LINEAR REGRESSION FUNCTION

linreg_fit_normeqn <- function(dataframe,x,y) {
  set.seed(124)
  require(ggplot2)
  N<-nrow(data)
  D<-(ncol(data)-1)

  #Simple Linear
  if (D==1){
    x<-as.matrix(x)
    y<-as.vector(y)
    dummy<- matrix(1, N)
    X<-cbind(dummy,x)
    w<-solve(t(X)%*%X,t(X)%*%y)
    yhat<-X%*%w

    residuals<-y-yhat
    R2<- 1-sum((y-yhat)^2)/sum((y-mean(y))^2)
    df<-data.frame(x=x,y=y)
    plt<-ggplot(df,aes(x=x))+geom_point(aes(y=y))+geom_line(aes(y=yhat))
    fit<-list(Weights=w,Rsquared=R2,Residuals=summary(residuals),Plot=plt)


  }
#Multiple Linear
  else{

    x<-as.matrix(x)
    y<-as.vector(y)
    dummy<- matrix(1, N)
    X<-cbind(dummy,x)
    w<-solve(t(X)%*%X,t(X)%*%y)
    yhat<-X%*%w

    residuals<-y-yhat
    R2<- 1-sum((y-yhat)^2)/sum((y-mean(y))^2)
    fit<-list(Weights=w,Rsquared=R2,Residual=summary(residuals))

  }

  return(fit)

}



#GRADENT DESCENT ALGORITHM

linregfit_grad_desc <- function(data,x,y,eta=0.001,epoch=2000,lmda_l2=0,lmda_l1=0) {
   set.seed(124)

   N<-nrow(data)
   D<-ncol(x)

   dummy<-matrix(1,N)
   X<-as.matrix(x)
   y<-as.matrix(y)

   X<-cbind(dummy,X)
   w<-as.matrix(rnorm(D+1))

   J=character()
   for(i in 0:epoch){
     yhat<-X%*%w
     J[i]<-(t(y-yhat)%*%(y-yhat))+lmda_l2*(t(w)%*%w)+lmda_l1*sum(abs(w))
     w= w-eta*(t(X)%*%(yhat-y)+lmda_l2*w+ lmda_l1*sign(w))
   }
   yhat<-X %*% w
   plt<-plot(J)
   R2<- 1-sum((y-yhat)^2)/sum((y-mean(y))^2)
   relerror <- abs((y - yhat) / y)
   MRE<-median(relerror)
   result<-list(Weights=w,MRE=MRE,Rsquared=R2,Plotof_J=plt)


   return(result)
}



#PREDICT FUNCTION

linreg_perdict<-function(fit,x,y){
  D<-length(fit$Weights)
  #Simple linear Predict
  if(D==2){
    yhat<-fit$Weights[1]+fit$Weights[2]*x
    residuals<-y-yhat
    R2<- 1-sum((y-yhat)^2)/sum((y-mean(y))^2)
    df<-data.frame(x=x,y=y)
    plt<-ggplot(df,aes(x=x))+geom_point(aes(y=y))+geom_line(aes(y=yhat))
    predict<-list(Rsquared=R2,Residuals=summary(residuals),Plot=plt)


  }
  # Multiliear Predict Function
  else{
    w<-fit$Weights
    N<-nrow(x)
    dummy<-matrix(1,N)
    X<-cbind(dummy,x)
    yhat<-X%*%w

    residuals<-y-yhat
    R2<- 1-sum((y-yhat)^2)/sum((y-mean(y))^2)
    relerror <- abs((y - yhat) / y)
    MRE<-median(relerror)
    predict<-list(MRE=MRE,Rsquared=R2,Residual=summary(residuals))




  }
  return(predict)



}




