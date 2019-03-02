# Geo Tool

A simple tool implemetation on gdal. Now supporting grid clip, clip with shape file, mask with shape file and merge tif. It has been tailored for computer vision in remote sensing. Help you construct your pipline easily! 
**Also see an visiliazed UI on Baidu Pan: https://pan.baidu.com/s/1FfDVzu3mlM9wItu7KVTl3A Fetch code: 2333**

## Getting Started

<div align=center><img width="150" height="150" src="src/ico/title.png"/></div>

### Prerequisites



```
GDAL
numpy >= 1.11.1
```

### Installing

```
pip install geotool
```

## Usage Example
```python
from geotool.tif_process import GeoTif, define_ref_predict
from geotool.tif_merge import run_merge
tif = GeoTif('xx.tif')
# if you want to clip tif with grid reserved geo reference
tif.clip_tif_with_grid(512, 'out_dir')
# if you want to clip tif with shape file
tif.clip_tif_with_shapefile('shapefile.shp', 'out_dir')
# if you want to mask tif with shape file
tif.mask_tif_with_shapefile('shapefile.shp', 'save_path.tif')
# After you get your model results(pngs or tifs), you can redefine their reference
# with you clipped before, notice they must match names
define_ref_predict('cipped_before_dir', 'your_result_dir', 'save_dir')
# if you want to merge tifs
run_merge('tif_dir', 'save_path.tif')
```

## Authors

* **Kingdrone** - *Initial work* - [geotool](https://github.com/Kindron/geotool)


