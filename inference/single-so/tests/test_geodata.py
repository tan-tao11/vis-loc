import tempfile
import unittest
from pathlib import Path
import sys
import warnings
import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from geodata import GeoReference, world_transform

class GeoTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.crs=CRS.from_epsg(32645)
        self.t=Affine(.5,0,500000,0,-.5,3000000)
    def tearDown(self):self.tmp.cleanup()
    def write(self,name,values,t,sidecars=False):
        path=self.root/name
        if values.ndim==2:values=values[None]
        kw={} if sidecars else dict(crs=self.crs.to_wkt(),transform=t)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with rasterio.open(path,'w',driver='GTiff',height=values.shape[1],width=values.shape[2],
                count=values.shape[0],dtype=values.dtype,compress='deflate',tiled=True,blockxsize=16,blockysize=16,**kw) as dst:dst.write(values)
        if sidecars:
            c,f=t*(.5,.5)
            path.with_suffix('.tfw').write_text('\n'.join(map(str,[t.a,t.d,t.b,t.e,c,f])))
            path.with_suffix('.prj').write_text(self.crs.to_wkt(version='WKT1_ESRI'))
        return path
    def ref(self,**kwargs):
        image=self.write('map.tif',np.full((3,64,64),120,np.uint8),self.t,True)
        dsm=self.write('dsm.tif',np.full((64,64),80,np.float32),self.t,True)
        return dict(image=str(image),dsm=str(dsm),window_m=16,**kwargs)
    def test_tfw_pixel_centres_and_rotation(self):
        p=self.root/'x.tfw';p.write_text('2\n0.1\n0.2\n-3\n500000\n3000000\n')
        t=world_transform(p)
        np.testing.assert_allclose(t*(.5,.5),(500000,3000000))
        np.testing.assert_allclose(t*(1.5,.5),(500002,3000000.1))
    def test_plain_tiff_sidecars_compressed_and_height(self):
        ref=GeoReference(self.ref())
        self.assertEqual(ref.shape,(64,64));self.assertEqual((ref.x0,ref.y0),(500000.25,2999999.75))
        lon,lat=ref.inverse.transform(500016,2999984)
        view=ref.for_frame(dict(lon=lon,lat=lat))
        self.assertEqual(view.image.shape,(32,32,3));np.testing.assert_allclose(view.dsm,80)
        self.assertAlmostEqual(view.agl(dict(lon=lon,lat=lat,altitude=180),{}),100)
        xyz=dict(X=5,Y=6,Z=100)
        out=view.local_to_geographic(xyz);x,y=view.forward.transform(*out[:2])
        self.assertAlmostEqual(x,view.x0+5,places=5);self.assertAlmostEqual(y,view.y0-6,places=5)
        self.assertEqual(out[2],180);ref.close()
    def test_different_grid_resolution(self):
        cfg=self.ref()
        cfg['dsm']=str(self.write('coarse.tif',np.full((32,32),55,np.float32),Affine(1,0,500000,0,-1,3000000)))
        r=GeoReference(cfg);v=r.read_window(0,32,0,32)
        self.assertEqual(v.image.shape,(32,32,3));np.testing.assert_allclose(v.dsm,55);r.close()
    def test_crs_conflict_fails(self):
        cfg=self.ref(crs='EPSG:32649')
        with self.assertRaisesRegex(ValueError,'disagrees'):GeoReference(cfg)
    def test_rotated_raster_is_rectified(self):
        cfg=self.ref();t=self.t*Affine.rotation(12)
        cfg['dsm']=str(self.write('rotated.tif',np.full((64,64),80,np.float32),t,True))
        r=GeoReference(cfg);v=r.read_window(0,r.shape[0],0,r.shape[1])
        self.assertTrue(np.isfinite(v.dsm).any());np.testing.assert_allclose(v.dsm[np.isfinite(v.dsm)],80);r.close()
    def test_missing_georeference_fails(self):
        cfg=self.ref();p=Path(cfg['image']);p.with_suffix('.tfw').unlink()
        with self.assertRaisesRegex(ValueError,'no geotransform'):GeoReference(cfg)
    def test_uint16_requires_explicit_mapping(self):
        cfg=self.ref();cfg['image']=str(self.write('wide.tif',np.full((3,64,64),32768,np.uint16),self.t))
        with self.assertRaisesRegex(ValueError,'image_value_range'):GeoReference(cfg)
        cfg['image_value_range']=[0,65535];r=GeoReference(cfg);v=r.read_window(0,32,0,32)
        np.testing.assert_allclose(v.image,127);r.close()
    def test_nodata_rejected(self):
        cfg=self.ref(nodata=80)
        r=GeoReference(cfg)
        with self.assertRaisesRegex(ValueError,'no overlapping'):r.read_window(0,32,0,32)
        r.close()

if __name__=='__main__':unittest.main(verbosity=2)
