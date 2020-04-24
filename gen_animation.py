import imageio

letter_dir = 'try'
syn_images = []
for letter in range(21):
    # plt.imshow(latents_letters[letter]*255)
    # plt.show()
    img_path = letter_dir+'/'+str(letter)+'.png'
    # r = np.reshape(latents_letters[letter], (64,64))
    # r = resize(latents_letters[letter], (64,64))
    # cv2.imwrite(img_path, r*255)
    syn_images.append(imageio.imread(img_path))
print(len(syn_images))
imageio.mimsave(letter_dir + '/generation_animation.gif', syn_images, fps=120, duration = 0.1)