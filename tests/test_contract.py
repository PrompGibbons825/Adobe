import unittest,copy,os,random,time
os.environ['ORIGIN_PIXELS']='0'
from engine.resize import resize,diagnose
from engine.objective import violations

def text(i,x,y,w,h,fs=28):
    return dict(id=i,type='text',x=x,y=y,width=w,height=h,font_size=fs,content='Something worth remembering',font_family='Solstice Sans',color='rgba(20,20,20,1)',z_index=2)
def source():
    return {'canvas':{'width':800,'height':600},'elements':[
        dict(id='background',type='shape',x=0,y=0,width=800,height=600,shape_type='rectangle',fill_color='rgba(250,250,250,1)',z_index=0),
        text('a',80,120,640,80,40),text('b',80,230,640,100,22),
        dict(id='image',type='image',x=220,y=370,width=360,height=180,asset_id='solstice_bottle.png',fit='contain',z_index=1)]}

class Contract(unittest.TestCase):
    def test_ids_copy_assets_bounds(self):
        src=source()
        for target in ({'width':728,'height':90},{'width':1080,'height':1920},{'width':300,'height':250}):
            out=resize(src,target);self.assertEqual(violations(src,out),[])
            self.assertEqual(src,source())
    def test_identity(self):
        src=source();self.assertEqual(resize(src,src['canvas']),src)
    def test_deterministic(self):
        src=source();t={'width':600,'height':800};self.assertEqual(resize(src,t),resize(src,t))
    def test_does_not_hide(self):
        src=source();out=resize(src,{'width':320,'height':50})
        self.assertTrue(all(e.get('opacity',1)>0 for e in out['elements']))
    def test_nested_shapes_do_not_explode(self):
        src=source()
        for i in range(90):src['elements'].append(dict(id='q'+str(i),type='shape',x=5+i,y=5+i,width=700-i*2,height=500-i*2,shape_type='rectangle',fill_color='rgba(240,230,220,1)',opacity=.1,z_index=0))
        t=time.process_time();out=resize(src,{'width':1200,'height':300});self.assertEqual(violations(src,out),[]);self.assertLess(time.process_time()-t,10)
    def test_unicode_does_not_drop_copy(self):
        src=source();src['elements'][1]['content']='Ćao — Καλημέρα 你好 مرحبا';out=resize(src,{'width':500,'height':500});self.assertEqual(out['elements'][1]['content'],src['elements'][1]['content'])
    def test_rotation_is_inside(self):
        src=source();src['elements'][1]['rotation']=35;out=resize(src,{'width':300,'height':250});self.assertEqual(violations(src,out),[])
    def test_translucent_decoration_is_not_a_container(self):
        from engine.scene import infer
        src=source();src['elements'].append(dict(id='veil',type='shape',x=0,y=0,width=600,height=500,shape_type='ellipse',fill_color='rgba(255,255,255,1)',opacity=.15,z_index=0))
        scene=infer(src);self.assertIn(4,scene.decor);self.assertTrue(all(4 not in q.ids for q in scene.islands))
    def test_element_order_and_ids_do_not_drive_layout(self):
        src=source();target={'width':600,'height':800};a=resize(src,target)
        other=copy.deepcopy(src);other['elements'].reverse();b=resize(other,target)
        self.assertEqual({e['id']:e for e in a['elements']},{e['id']:e for e in b['elements']})
        rename={e['id']:'new-'+str(i) for i,e in enumerate(src['elements'])}
        other=copy.deepcopy(src)
        for e in other['elements']:e['id']=rename[e['id']]
        b=resize(other,target)
        for e in b['elements']:e['id']=next(k for k,v in rename.items() if v==e['id'])
        self.assertEqual(a,b)
    def test_adjacent_fragments_are_one_typographic_island(self):
        from engine.scene import infer
        src={'canvas':{'width':1000,'height':500},'elements':[text('left',100,200,200,50,30),text('right',320,200,200,50,30)]}
        src['elements'][0]['content']='HELLO';src['elements'][1]['content']='WORLD'
        # Use the actual glyph adjacency, not generously allocated text boxes.
        from engine.typefit import footprint
        src['elements'][1]['x']=footprint(src['elements'][0]).r+10
        scene=infer(src);self.assertEqual(len(scene.islands),1)
    def test_public_entry_matches_experiment_portfolio(self):
        from engine.resize import diagnose_all
        src=source();target={'width':500,'height':800};methods,_=diagnose_all(src,target,False)
        self.assertEqual(resize(src,target),methods['portfolio']['output'])
    def test_rotation_uses_text_box_center(self):
        from engine.typefit import footprint,painted
        e=text('r',10,20,300,100,20);e['content']='Hi';e['rotation']=180
        ink=footprint(e);actual=painted(e)
        self.assertAlmostEqual(actual.x,2*(e['x']+e['width']/2)-ink.r)
        self.assertAlmostEqual(actual.y,2*(e['y']+e['height']/2)-ink.b)
    def test_all_algorithms_work(self):
        for method in ('elastic','hierarchy','search','portfolio'):
            best,info=diagnose(source(),{'width':400,'height':700},method,False)
            self.assertEqual(violations(source(),best['output']),[]);self.assertGreater(info['generated'],1)
if __name__=='__main__':unittest.main()
