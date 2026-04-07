from dustmaps.config import config
config['data_dir'] = '/pscratch/sd/p/pratsosa/dust_maps'

import dustmaps.sfd
dustmaps.sfd.fetch()