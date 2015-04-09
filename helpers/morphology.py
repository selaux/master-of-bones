from skimage.measure import label, regionprops

def find_largest_label(possible_bone_pixels):
    labels = label(possible_bone_pixels, neighbors=8, background=0)
    labels = labels + 1

    label_props = regionprops(labels)
    label_with_max_area = label_props[0]
    for labelprop in label_props:
        if label_with_max_area.area <= labelprop.area:
            label_with_max_area = labelprop
    return labels == label_with_max_area.label
