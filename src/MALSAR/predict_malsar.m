function yscoredi = predict_malsar(Tmodel,Xdi,di)
W = Tmodel.W;
c = Tmodel.c;
yscoredi = Xdi*W(:,di)+c(di);