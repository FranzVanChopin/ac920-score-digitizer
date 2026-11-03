"""Run from repo root after host tests and building capture_denoise.dll.
Uses local copies of three user photos; never writes to the SD card.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def main():
    folder = Path('build/camera_noise_20261028')
    dll_path = Path('build/camera_capture_tests/capture_denoise.dll')
    lib = ctypes.CDLL(str(dll_path.resolve()))
    run = lib.capture_denoise_bgr
    run.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    reports = []
    sheet = Image.new('RGB', (1024, 2304))
    for i, name in enumerate(('000002', '000004', '000006')):
        source = folder / (name + '.bmp')
        rgb = np.array(Image.open(source))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        workspace = ctypes.create_string_buffer(1024 * 1024)
        assert run(bgr.ctypes.data, bgr.shape[1], bgr.shape[0], bgr.strides[0],
                   workspace, None, None) == 0
        output = np.ascontiguousarray(bgr[:, :, ::-1])
        Image.fromarray(output).save(folder / (name + '_chroma.png'))
        report = {'file': str(source), 'sha256': digest(source), 'flat_roi_xywh': [1800, 1200, 500, 300]}
        for key, image in [('before', rgb), ('after', output)]:
            roi = image[1200:1500, 1800:2300].astype(float)
            y = (roi * [77, 150, 29]).sum(2) / 256
            report[key] = {'Cr_std': float((roi[:, :, 0] - y).std()),
                           'Cb_std': float((roi[:, :, 2] - y).std()), 'Y_std': float(y.std())}
        y1 = (rgb.astype(np.int32) * [77, 150, 29]).sum(2)
        y2 = (output.astype(np.int32) * [77, 150, 29]).sum(2)
        report['integer_luma_max_change'] = int(np.abs((y1 + 128)//256 - (y2 + 128)//256).max())
        assert report['integer_luma_max_change'] == 0
        for channel in ('Cr_std', 'Cb_std'):
            assert report['after'][channel] < report['before'][channel] * 0.55
        reports.append(report)
        for j, image in enumerate((rgb, output)):
            sheet.paste(Image.fromarray(image).crop((1772, 1180, 2284, 1948)), (512*j, 768*i))
            ImageDraw.Draw(sheet).text((512*j+8, 768*i+8), name + (' before' if j == 0 else ' after'),
                                      fill='black', stroke_width=1, stroke_fill='white')
    sheet.save(folder / 'denoise_comparison.png')
    decoded = []
    for name, bit_count, size in [('gray8', 8, 11633686), ('binary1', 1, 1457006)]:
        path = Path('build/camera_capture_tests') / (name + '.bmp')
        im = Image.open(path)
        data = np.asarray(im.convert('L'))
        yy, xx = np.indices(data.shape)
        expected = ((xx + yy) % 256).astype('uint8') if bit_count == 8 else np.where((xx + yy) % 3, 255, 0).astype('uint8')
        assert im.size == (4056, 2868) and np.array_equal(data, expected) and path.stat().st_size == size
        decoded.append({'file': str(path), 'mode': im.mode, 'bytes': size, 'sha256': digest(path)})
    report = {'passed': True, 'photos': reports, 'independent_bmp_decoder': decoded,
              'dll_sha256': digest(dll_path), 'source_sha256': digest('software/camera_capture/capture_denoise.c'),
              'test_sha256': digest(__file__), 'comparison': str(folder / 'denoise_comparison.png'),
              'board_test_performed': False,
              'scope': 'Existing corrected BMPs filtered by native C; no measurement of sensor changes or board save time.'}
    (folder / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS real-photo chroma: 3 frames, flat-ROI std reduced >45%, integer Y unchanged at every pixel')
    print('PASS Pillow independent native BMP decoder: Gray8 and Binary1 every pixel, size and orientation')

if __name__ == '__main__':
    main()
