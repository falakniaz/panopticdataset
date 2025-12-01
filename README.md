# UAV PanoSeg-A Large Scale Benchmark for Dual Modality Flying UAV Panoptic Segmentation

<p>We introduce a new UAV's based dataset designed to advance research in panoptic segmentation by integrating object recognition into the broader context of scene understanding. </p>

---

# Table of Contents

<a href= "#Overview">Overview</a><br>
<a href= "#Prerequisites">Prerequisites</a><br>
<a href= "#Getting Started">Getting Started</a><br>
<a href= "#Semantic composer">Semantic composer</a><br>
<a href= "#Instance Dataset Creation">Instance Dataset Creation</a><br>
<a href= "#Panoptic Dataset">Panoptic Dataset</a><br>
<a href= "#Dataset Structure">Dataset Structure</a><br>
<a href= "#Author & Contact">Author & Contact</a>
---
<h2><a class = "anchor" id="Overview"></a>Prerequisites</h2>

```
1. Install Python >= 3.13.5
2. Install Detectron2
3. Install os, re, cv2, argparse, random
4. Install numpy 
5. Install glob
6. install tqdm, typing
```

---
<h2><a class = "anchor" id="Getting Started"></a>Getting Started</h2>
Firstly we have created semantic dataset, which contained by multiple drones, after that we create instance and lastly Panoptic.

<h2><a class = "anchor" id="Semantic composer"></a>1. Semantic composer </h2>

```
python3 semanticcomposer_falak.py \
  --original_root /home/falak/UAV_dataset/combined_uav \
  --out_root /home/falak/UAV_dataset/combined_uav_multi \
  --images_dirname input --masks_dirname labels \
  --splits train,val,test \
  --recursive \
  --allow_flip \
  --min_copies 2 --max_copies 5 \
  --attempts_per_copy 60 \
  --scale_min 0.7 --scale_max 1.3 --rot_deg 18 \
  --object_id 1 \
  --verbose \
  --seed 42
```


---

<h2><a class = "anchor" id="Instance Dataset Creation"></a>2. Instance Creation  </h2>

```
python3 instance_builder_falak.py \
  --in_root  /home/falak/Full_revise_dataset \
  --out_root /home/falak/Full_revise_dataset_instance \
  --images_dirname input --masks_dirname labels \
  --mask_split_aliases images:labels \
  --splits images \
  --recursive \
  --copy_images \
  --min_area 20
```

---
<h2><a class = "anchor" id="Panoptic Dataset"></a>3. Panoptic Dataset </h2>


```

    python3 instance_to_panoptic.py \
  --image_root    /home/falak/Full_revise_dataset/semanticdataset \
  --instance_root /home/falak/Full_revise_dataset/instancedataset \
  --out_root      /home/falak/Full_revise_dataset/Panopticdataset \
  --images_dirname input \
  --instance_masks_dirname labels \
  --splits train,val,test \
  --recursive \
  --min_area 20 --id_multiplier 1000 \
  --category_id 1 --category_name uav_object

  ```
<h2><a class = "anchor" id = "Dataset Structure"></a>Dataset Structure </h2>

```
Dataset structure <br>
|
Panoptic dataset<br>
|__semantic<br>
    |__input<br>
       |__test
          |__20190925_111757_1_1....
          |_infrared
          |_visible
       |__train
         |__20190925_111757_1_1....
          |_infrared
          |_visible
       |__val
         |__20190925_111757_1_1....
          |_infrared
          |_visible
    |__labels<br>
           |__test
          |__20190925_111757_1_1....
                    |_infrared
                    |_visible
       |__train
         |__20190925_111757_1_1....
                   |_infrared
                   |_visible
       |__val
         |__20190925_111757_1_1....
                   |_infrared
                  |_visible

|__instance <br>
    |__input<br>
       |__test
          |__20190925_111757_1_1....
                    |_infrared
                    |_visible
       |__train
         |__20190925_111757_1_1....
                   |_infrared
                   |_visible
       |__val
         |__20190925_111757_1_1....
                   |_infrared
                  |_visible
    |__labels<br>
           |__test
          |__20190925_111757_1_1....
                    |_infrared
                    |_visible
       |__train
         |__20190925_111757_1_1....
                   |_infrared
                   |_visible
       |__val
         |__20190925_111757_1_1....
                   |_infrared
                  |_visible
|__panoptic<br>
    |__input<br>
       |__test
          |__20190925_111757_1_1....
                    |_infrared
                    |_visible
       |__train
         |__20190925_111757_1_1....
                   |_infrared
                  |_visible
       |__val
         |__20190925_111757_1_1....
                   |_infrared
                    |_visible
    |__labels<br>
           |__test
          |__20190925_111757_1_1....
                    |_infrared
                    |_visible
       |__train
         |__20190925_111757_1_1....
                   |_infrared
                   |_visible
       |__val
         |__20190925_111757_1_1....
                   |_infrared
                  |_visible
```

---


  <h2><a class = "anchor" id="Author & Contact"></a>Author & Contact </h2>
  Falak Niaz <br>
  Department of Electronic Engineering, Kookmin University Seoul South Korea <br>
  Email: <a>falakbhai300@gmail.com</a>
   
