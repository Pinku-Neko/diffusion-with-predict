from matplotlib import pyplot as plt
import matplotlib.animation as animation


def plot_image(image,is_gray):
    if is_gray:
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def animate_images(images,is_gray):
    fig = plt.figure()
    ims = []
    for i in range(len(images)):
        if is_gray:
            im = plt.imshow(images[i], cmap="gray", animated=True)
        else:
            im = plt.imshow(images[i], animated=True)
        ims.append([im])
    # add buffer
    for i in range(50):
        if is_gray:
            ims.append([plt.imshow(images[-1], cmap="gray", animated=True)])
        else:
            ims.append([plt.imshow(images[-1], animated=True)])

    animate = animation.ArtistAnimation(fig, ims, interval=25, blit=True, repeat_delay=1000)
    plt.show()

def compare_grayscale_animations(images_long, images_short, filename, is_gray):
    '''
    outputs 2 animation of images synchronized \n
    -images_long: images with more frames \n
    -images_short: images with fewer frames \n
    -postcondition: file "comparison.gif" saved. animation starts with images_long, so that they end at the same time
    '''
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a subplot with two axes (side by side)
    
    # Initialize two lists to store the animations
    ims = []
    frame_diff = len(images_long) - len(images_short)

    # Create animations for the first set of images
    for i in range(len(images_long)):
        if is_gray:
            im_long = axes[0].imshow(images_long[i], cmap="gray", animated=True)
        else:
            im_long = axes[0].imshow(images_long[i], animated=True)
        # if short animation does not start
        if i >= frame_diff:
            # use i + offset as index to grab image
            if is_gray:
                im_short = axes[1].imshow(images_short[i-frame_diff], cmap="gray", animated=True)
            else:
                im_short = axes[1].imshow(images_short[i-frame_diff], animated=True)
        else:
            # feed 1st image
            if is_gray:
                im_short = axes[1].imshow(images_short[0], cmap="gray", animated=True)
            else:
                im_short = axes[1].imshow(images_short[0], animated=True)
        ims.append([im_long,im_short])

    # repeat last frame to observe result
    if is_gray:
        last_frame_long = axes[0].imshow(images_long[-1], cmap="gray", animated=True)
        last_frame_short = axes[1].imshow(images_short[-1], cmap="gray", animated=True)
    else:
        last_frame_long = axes[0].imshow(images_long[-1], animated=True)
        last_frame_short = axes[1].imshow(images_short[-1], animated=True)
    for i in range(50):
        ims.append([last_frame_long,last_frame_short])

    # Create ArtistAnimations for both sets of images
    animate = animation.ArtistAnimation(fig, ims, interval=25, blit=True, repeat_delay=1000)
    animate.save(filename)

    plt.show()
