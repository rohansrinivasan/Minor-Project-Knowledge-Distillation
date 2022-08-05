#Distill Student to Teacher
distiller = Distiller1(student=model2, teacher=model1)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.CategoricalAccuracy()],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=3,
)
hist6=distiller.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=True, validation_data=(X_test, y_test)) #, epochs=epochs,callbacks=[es]
hist_arr6 = np.array([hist6.history['accuracy'],hist6.history['val_accuracy'],hist6.history['student_loss'],hist6.history['distillation_loss'],hist6.history['val_accuracy'],hist6.history['val_student_loss']])


hist3=distiller.fit(X_train, y_train, epochs=epochs) #, epochs=epochs,callbacks=[es]
# Test Accuracy
plt.figure(5)
plt.plot(hist_arr2[0])
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Accuracy
plt.figure(2)
plt.plot(hist_arr2[3])
plt.ylabel('Train Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)

# Test Loss
plt.figure(3)
plt.plot(hist_arr2[1])
plt.ylabel('Test Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Loss
plt.figure(4)
plt.plot(hist_arr2[4])
plt.ylabel('Train Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Distillation Loss
plt.figure(4)
plt.plot(hist_arr2[2])
plt.ylabel('Teacher over Student Distillation Loss')
plt.xlabel('Epoch')
plt.grid(True)