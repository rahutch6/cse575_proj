
roses = list(img_dir.glob('roses/*')) 
img = PIL.Image.open(str(roses[0]))
img.show()

class_names = train_ds.class_names 

num_classes = len(class_names) 

plt.figure(figsize=(10, 10)) 

for images, labels in train_ds.take(1): 
	for i in range(25): 
		ax = plt.subplot(5, 5, i + 1) 
		plt.imshow(images[i].numpy().astype("uint8")) 
		plt.title(class_names[labels[i]]) 
		plt.axis("off") 


#Accuracy 
acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy'] 

#loss 
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 

#epochs 
epochs_range = range(epochs) 

#Plotting graphs 
plt.figure(figsize=(8, 8)) 
plt.subplot(1, 2, 1) 
plt.plot(epochs_range, acc, label='Training Accuracy') 
plt.plot(epochs_range, val_acc, label='Validation Accuracy') 
plt.legend(loc='lower right') 
plt.title('Training and Validation Accuracy') 

plt.subplot(1, 2, 2) 
plt.plot(epochs_range, loss, label='Training Loss') 
plt.plot(epochs_range, val_loss, label='Validation Loss') 
plt.legend(loc='upper right') 
plt.title('Training and Validation Loss') 
plt.show() 
