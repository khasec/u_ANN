#imported libraries

import os.path

from google.oauth2.credentials import Credentials

from googleapiclient.discovery import build

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Importing the emails with help of the gmail api
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
hold=[]
hold2=[]
hold3=[]
k=0
creds = None

for z in range(2):
    
    token='token'+str(z)+'.json'
    
    if os.path.exists(token):
        
        creds = Credentials.from_authorized_user_file(token, SCOPES)

    
    # Call the Gmail API
    service = build('gmail', 'v1', credentials=creds)
    results = service.users().messages().list(userId='me',labelIds=['INBOX']).execute()
    messages = results.get('messages', [])
    messages = [service.users().messages().get(userId='me', id=msg['id'],format='raw').execute() for msg in messages]


    test= pd.read_csv("mail.csv")
        
        
    a = list(test['Message'])
    
    
    for i in messages:

        data = {
                'Category': [z],
                'Message': [i['snippet']],   
                }
        
        if data['Message'][0] not in a:

            df = pd.DataFrame(data)
            hold2.append(data['Message'][0])
            hold3.append(k)

    creds=[]
    k=+1



#formatting the data and prepering it for the model
data= pd.read_csv("mail.csv")

X=data["Message"]
y=data["Category"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=48)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_train = feature_extraction.fit_transform(X_train)

X_test = feature_extraction.transform(X_test)

X_train=X_train.toarray()
X_test=X_test.toarray()

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# define the keras model and building the neural network model
model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, Y_train, shuffle=True, epochs=20, batch_size=10)
accuracy_loss=model.evaluate(X_test, Y_test)


Xt_train = hold2
Input_test = feature_extraction.transform(Xt_train)
Input_test=Input_test.toarray()

#test the model on the new data and adding it to the data file. 

predictions = model.predict(Input_test)
z=0
news=0
newstext=[]
game=0
gametext=[]
print('-----------------------------------------------')
for i in predictions:
  z=z+1
  print(f'{i[0]} for {Xt_train[z-1]}')
  
  
  if i <= 0.05:
    data = {
    'Category': [0],
    'Message': [Xt_train[z-1]],
    'real': [hold3[z-1]]   
    }
    df = pd.DataFrame(data)
    if len(Xt_train[z-1]) > 50:
      df.to_csv("mail.csv", mode='a', index=False, header=False)
      game=game+1
      gametext.append(data)
  elif i >= 0.95:
    data = {
    'Category': [1],
    'Message': [Xt_train[z-1]],
    'real': [hold3[z-1]]      
    }
    df = pd.DataFrame(data)
    if len(Xt_train[z-1]) > 50:
      df.to_csv("mail.csv", mode='a', index=False, header=False)
      newstext.append(data)
      news=news+1


print('----------------Model_Evaluation-----------------')
print('curent thresouhold: 0.05 and 0.95')
print('\n')
print('Added mails from r/gaming:')
print('\n')
for i in gametext:
  print(i['Message'])
  print('\n')
print('Added mails fro r/worldnews:')
for i in newstext:
  print(i['Message'])
  print('\n')
print('\n')
print('\n')
print(f'total number of added emails to list {news+game}')
print('\n')
print('\n')
print(f'total number of r/gaming emails to list {game}')
print(f'total number of r/worldnews emails to list {news}')
print(f'total number of emails to list {len(X)+game+news}')
print('\n')

print('model evaluation:')
print(f'model accuracy {accuracy_loss[1]}')
print(f'model Loss {accuracy_loss[0]}')
print('-----------------------------------------------')

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()






        
