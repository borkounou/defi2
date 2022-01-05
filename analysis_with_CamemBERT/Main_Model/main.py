from main_trainer import trainer_local
from predictor import predictor_local


if __name__ =="__main__":
    predict = True
    # train the model
    '''
    Comment the below line if you do not want to train or you have already trained
    '''
    print("Start training...")
    trainer_local()
    print("Training completed!")
    
    # predict
    if predict:
        print("Start predicting...")
        predictor_local()
        print("Prediction completed!")
    
