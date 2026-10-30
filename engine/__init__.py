import os as _os
if 'ORIGIN_ASSETS' not in _os.environ and _os.environ.get('CHAMELEON_BRAND_KIT'):
    _os.environ['ORIGIN_ASSETS'] = _os.path.join(_os.environ['CHAMELEON_BRAND_KIT'], 'assets')
