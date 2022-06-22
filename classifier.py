import numpy as np
import os
from PIL import Image
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D, Dense
#from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from datasetcreator import DatasetCreator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchsummary import summary
from dataset import X_train, X_test, y_train, y_test


'''model = Sequential()

model.add(BatchNormalization()) # Just to make sure.
model.add(Conv2D(6, kernel_size=5, activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=1, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(7744, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.4))
model.add(Dense(120, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.4))
model.add(Dense(84, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax', kernel_initializer='glorot_uniform'))

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=10, epochs=1000, verbose=1) # The model's performance gets lower after too many epochs(accuracy drops from ~95% to 55%)

model.save_weights('image_classifier')

#model.load_weights('image_classifier') # After a good training, simply load the weights.


y_pred = model.predict(X_test, batch_size=2, verbose=1)

for i in range(X_test.shape[0]):
    predicted = classes[np.argmax(y_pred[i])]
    real = classes[np.argmax(y_test[i])]
    imagem = (X_test[i]+1)*0.5
    plt.imshow(imagem)
    plt.title(f'Predicted outcome: {predicted}\nReal outcome: {real}')
    plt.show()

# Finally, labeling the remaining images using the Neural Network itself.

remains = data[2050:]
y_rest = model.predict(remains, batch_size=2, verbose=1)

y = np.concatenate((y_train, y_test, y_rest))'''

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 1, 4, 1, 0, bias=False)
        # Level 2:
        '''self.conv1 = nn.Conv2d(3, 30, 6, 2, 2, bias=False)
        self.LeakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.4, inplace=False) # Perhaps consider using Dropout2D? Or even 3D?
        self.conv2 = nn.Conv2d(30, 1, 4, 1, 0, bias=False)'''
        # And so on
        '''self.batchnorm2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)'''
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        output = self.sigmoid(x)
        # Level 2:
        '''x = self.conv1(input)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        # Further and further
        x = self.batchnorm2(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.LeakyRelu(x)
        x = self.dropout(x)
        x = self.conv5(x)'''
        output = self.sigmoid(x)

        return output

# Create the Discriminator
netD = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

def train(data=None, epochs=1000, batch_size=6,loss=nn.BCELoss(), optimizerD=optimizerD, optimizerG=optimizerG, save_point=100, checkpoint=5000, model_name='model'):
    if os.path.isfile(f'discriminator_{model_name}.pth'):
        try:
            netD.load_state_dict(torch.load(f'discriminator_{model_name}.pth')) # Loading checkpoint
        
        except RuntimeError:
            previous_discriminator = torch.load(f'discriminator_{model_name}.pth') # Loading previous model's weights
            current_discriminator = netD.state_dict()
            desired_shape = current_discriminator['conv1.weight'].size()
            previous_shape = previous_discriminator['conv1.weight'].size()
            zeros = torch.zeros(desired_shape[0], desired_shape[1], previous_shape[2], previous_shape[3]).to(device)
            weights = torch.add(zeros, previous_discriminator['conv1.weight'])
            weights = upsampler(weights)
            current_discriminator['conv1.weight'] = weights

    print("Weights Updated!")

    for epoch in range(epochs):
        netD.zero_grad()
        b_size = batch_size
        real_cpu = data[np.random.randint(0, data.shape[0], size=batch_size), :, :, :].to(device)
        label = torch.full((real_cpu.shape[0],), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1) # Gera um tensor com shape (batch_size,)
        errD = loss(output, label)
        errD.backward()
        D_x = output.mean().item()
        # Update D
        optimizerD.step()
        # Update scheduler
        schedulerD.step()
        

        best_disc_loss = float('inf')
        discriminator_loss = errD.item()
                
        if discriminator_loss < best_disc_loss:
            best_disc_loss = discriminator_loss
            best_discriminator_parameters = netD.state_dict()

        # Output training stats
        if epoch % checkpoint == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs,
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            #print(f'Discriminator last LR: {schedulerD.get_last_lr()}')
            
            torch.save(best_discriminator_parameters, f'discriminator_{model_name}.pth')
            print("Model saved!")


# And filtering images that we don't want in our dataset. Unfortunately, numpy.delete() returns a flattened array, so it can't be used.

filtered_data = []
filtered_labels = []

for i in range(data.shape[0]):
    label = classes[np.argmax(y[i])]
    if label != "Undesired?":
      filtered_data.append(data[i])
      filtered_labels.append(y[i])

filtered_data, filtered_labels = np.array(filtered_data), np.array(filtered_labels)

print(filtered_data.shape, filtered_labels.shape)

# Checking if the Neural Network isn't too crazy on its labeling...

for i in range(10):
    image = filtered_data[7515-1-i]
    image = (image+1.0)*0.5
    plt.imshow(image)
    plt.title(f"Predicted Label: {classes[np.argmax(filtered_labels[7515-1-i])]}")
    plt.show()


# Saving the filtered data:

np.save("filtered_images", filtered_data)
