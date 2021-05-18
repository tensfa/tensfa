 #Create arrays
nn = [2, 4, 6, 8, 12, 10, 8, 2]
labels = []
words = []
docs_x = []
docs_y = []

with open("intents.json") as file:
    data = json.load(file)

for intent in data["intents"]:
    for pattern in intent["patterns"]:
          pattern = intent["tag"]
    for response in intent["responses"]:
          response = intent["responses"]
    for tags in intent["tag"]:
          tags = intent["tag"]

    #Add tags to labels[]
    if intent["tag"] not in labels:
      labels.append(intent["tag"])

    #Add patterns to docs_y[]
    if intent["patterns"] not in words:
      docs_y.append(intent["patterns"])

def create_class():
  y = np.array(labels)
  x = np.array(words)

  classes = np.unique(y)
  nClasses = len(classes)

  print("Number of classes: " , nClasses)
  print("Classes: " , classes)

def create_training_data():
  training_data = np.ones(np.shape(docs_y))
  target_data = np.ones(np.shape(labels))

  print("Training Data Shape: " , training_data)
  print("Target Data Shape: " , target_data)

#Create training shapes:
create_training_data()

def model():
  INIT_LR = 1e-3
  epochs = 6
  batch_size = 64

  model = kr.Sequential()

  model.add(kr.layers.Dense(nn[1], activation='relu'))
  model.add(kr.layers.Dense(nn[2], activation='relu'))
  model.add(kr.layers.Dense(nn[3], activation='relu'))
  model.add(kr.layers.Dense(nn[4], activation='relu'))
  model.add(kr.layers.Dense(nn[5], activation='relu'))
  model.add(kr.layers.Dense(nn[6], activation='relu'))
  model.add(kr.layers.Dense(nn[7], activation='sigmoid'))

  model.compile(loss=kr.losses.categorical_crossentropy, optimizer=kr.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])

  #Training
  model.fit(training_data , target_data , epochs=100)

#Call fns
create_class()
model()