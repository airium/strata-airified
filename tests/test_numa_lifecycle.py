import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/strata"
FILE = SOURCE / "serve/numa_lifecycle.py"
spec = importlib.util.spec_from_file_location("numa_policy_test", FILE)
policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)


class Topology(unittest.TestCase):
    def fixture(self, *, nodes=4, sockets=1, vendor="AuthenticAMD", model="AMD EPYC 7B13", restricted=False):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); root = Path(tmp.name)
        def put(name, data):
            p=root/name.lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
        put('/proc/cpuinfo',f'vendor_id : {vendor}\nmodel name : {model}\n')
        put('/sys/devices/system/cpu/online',f'0-{nodes*2-1}\n')
        put('/proc/self/cgroup','0::/service/leaf\n')
        put('/proc/self/status',f'Mems_allowed_list:\t0-{nodes-1}\n')
        put('/proc/self/mountinfo','37 28 0:31 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n')
        (root/'sys/fs/cgroup/service/leaf').mkdir(parents=True)
        put('/sys/fs/cgroup/cpuset.cpus.effective',f'0-{nodes*2-1}\n')
        if restricted:put('/sys/fs/cgroup/service/cpuset.cpus.effective','0-1\n')
        for n in range(nodes):
            put(f'/sys/devices/system/node/node{n}/cpulist',f'{n*2}-{n*2+1}\n')
            put(f'/sys/devices/system/node/node{n}/meminfo',f'Node {n} MemTotal: 1000000 kB\n')
            for c in range(n*2,n*2+2):put(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id',str(n*sockets//nodes))
        return root

    def test_effective_nps4_and_inherited_cgroup_controller(self):
        r=policy.topology(self.fixture());self.assertTrue(r['eligible']);self.assertEqual(r['cpus'],list(range(8)))

    def test_four_nodes_are_not_enough(self):
        for settings in [dict(sockets=2),dict(nodes=2),dict(nodes=8),dict(vendor='GenuineIntel',model='Xeon'),dict(model='AMD Ryzen Threadripper'),dict(restricted=True)]:
            with self.subTest(settings=settings):self.assertFalse(policy.topology(self.fixture(**settings))['eligible'])

    def test_memoryless_asymmetric_and_unknown_topologies_stay_off(self):
        for size in [0,400000]:
            root=self.fixture();(root/'sys/devices/system/node/node0/meminfo').write_text(f'Node 0 MemTotal: {size} kB\n')
            self.assertFalse(policy.topology(root)['eligible'])
        with tempfile.TemporaryDirectory() as root:self.assertFalse(policy.topology(Path(root))['eligible'])
        self.assertFalse(policy.topology(self.fixture(),platform='darwin')['eligible'])
        root=self.fixture();(root/'proc/self/status').write_text('Mems_allowed_list:\t0-1\n')
        self.assertFalse(policy.topology(root)['eligible'])

    def test_verified_native_support_and_optouts(self):
        root=self.fixture();binary=root/'build/strata';binary.parent.mkdir();binary.write_bytes(b'fixture engine')
        data={'patches':[{'path':'patches/'+x} for x in policy.REQUIRED_PATCHES],
              'payloads':{'strata':{'bytes':binary.stat().st_size,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}}}
        (binary.parent/'BUILD.json').write_text(json.dumps(data))
        self.assertTrue(policy.configuration(str(binary),{},'auto','unused',root=root)['enabled'])
        self.assertFalse(policy.configuration(str(binary),{},'off','unused',root=root)['enabled'])
        self.assertFalse(policy.configuration(str(binary),{'STRATA_NUMA_SPLIT':'0'},'auto','unused',root=root)['enabled'])
        binary.write_bytes(b'changed engine');self.assertFalse(policy.configuration(str(binary),{},'auto','unused',root=root)['enabled'])
        with self.assertRaises(ValueError):policy.configuration(str(binary),{},'bad','unused',root=root)

    def test_pid_reuse_is_not_touched(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'lease.json';row={'engine_pid':4242,'start_ticks':'old','daemon':[7,'daemon']}
            with mock.patch.object(policy,'identity',return_value=('new','S')),mock.patch.object(policy,'control') as command:
                policy.cleanup({'record':str(path),'pidfile':'unused','command':['unused']},row)
            command.assert_not_called();self.assertEqual(row['status'],'pid-reused')

    @unittest.skipUnless(sys.platform.startswith('linux'),'Linux file descriptors')
    def test_failed_bootstrap_popen_closes_both_handshake_fds(self):
        with tempfile.TemporaryDirectory() as root:
            env={'STRATA_NUMA_SPLIT':'4',policy.STATE_DIR:root}
            before=set(os.listdir('/proc/self/fd'))
            with mock.patch.object(policy,'topology',return_value={'eligible':True,'cpus':sorted(os.sched_getaffinity(0))}),mock.patch.object(policy.subprocess,'Popen',side_effect=OSError('test spawn failure')):
                with self.assertRaises(OSError):policy.guarded_popen([sys.executable],env=env,_controller={'command':['unused'],'pidfile':root+'/absent.pid'})
            self.assertEqual(set(os.listdir('/proc/self/fd')),before)


SERVER = r'''
import ctypes,json,os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from serve import lifeline,numa_lifecycle as n
folder=Path(sys.argv[2]);mode=sys.argv[3]
allowed=sorted(n.effective_cpus());one={allowed[0]}
if mode!='explicit-mask':os.sched_setaffinity(0,one)
# Topology is independently fixture-tested above. These protocol tests must also
# run on a non-EPYC Linux CI host, while using its actual cgroup/affinity rules.
n.topology=lambda:{'eligible':True,'reason':'process fixture','cpus':allowed}
original=n.guarded_popen
def guarded(args,**kw):
 return original(args,_controller={'pidfile':str(folder/'daemon.pid'),'command':[sys.executable,str(folder/'control.py'),str(folder/'commands.jsonl'),'deny' if mode=='deny' else 'allow']},**kw)
n.guarded_popen=guarded
env={**os.environ,'STRATA_NUMA_SPLIT':'4',n.GUARD:'1',n.STATE_DIR:str(folder/'state')}
engine_code="import json,os,sys,time;from pathlib import Path;Path(sys.argv[1]).write_text(str(time.monotonic()));print(json.dumps({'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0))}),flush=True);sys.stdin.readline()"
kwargs={'stdin':__import__('subprocess').PIPE,'stdout':__import__('subprocess').PIPE,'stderr':__import__('subprocess').PIPE,'text':True,'env':env}
if mode=='explicit-mask':kwargs['preexec_fn']=lambda:os.sched_setaffinity(0,one)
args=[sys.executable,'-c',engine_code,str(folder/'executed')]
if mode=='exec-fail':args=[str(folder/'does-not-exist')]
try:
 child=lifeline.popen(args,'vision' if mode=='vision' else 'engine',**kwargs)
 reply=json.loads(child.stdout.readline());reply.update(server=os.getpid(),child=child.pid,parent_affinity=sorted(os.sched_getaffinity(0)))
 print(json.dumps(reply),flush=True)
 command=sys.stdin.readline().strip()
 if command in ('quit','restart'):
  child.stdin.write('QUIT\n');child.stdin.flush();child.wait(timeout=10);n.release(child)
 if command=='restart':
  child=lifeline.popen(args,'engine',**kwargs);reply=json.loads(child.stdout.readline());reply['child']=child.pid;print(json.dumps(reply),flush=True)
  sys.stdin.readline();child.stdin.write('QUIT\n');child.stdin.flush();child.wait(timeout=10);n.release(child)
except OSError as error:print(json.dumps({'error':str(error)}),flush=True)
'''


@unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os,'sched_setaffinity'),'Linux process ownership')
class ProcessLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name);self.servers=[];self.native=[]
        control="""import json,sys,time
from pathlib import Path
path=Path(sys.argv[1]);mode=sys.argv[2];args=sys.argv[3:]
if args[0] in ['-x','-r']:
 with path.open('a') as f:f.write(json.dumps({'action':args[0],'pid':int(args[1]),'time':time.monotonic(),'args':args})+'\\n')
 if mode=='deny' and args[0]=='-x':raise SystemExit(1)
print('0')
"""
        (self.folder/'control.py').write_text(control);(self.folder/'server.py').write_text(SERVER)
        self.daemon=subprocess.Popen([sys.executable,'-c',"import ctypes,os,time;ctypes.CDLL(None).prctl(15,b'numad',0,0,0);print(os.getpid(),flush=True);time.sleep(600)"],stdout=subprocess.PIPE,text=True)
        pid=int(self.daemon.stdout.readline());(self.folder/'daemon.pid').write_text(str(pid))
        self.addCleanup(self.clean)

    def clean(self):
        for p in self.servers:
            if p.poll() is None:p.kill()
            p.wait(timeout=10)
            for stream in [p.stdin,p.stdout,p.stderr]:
                if stream:stream.close()
        for pid in self.native:
            ident=policy.identity(pid)
            if ident and ident[1]!='Z':
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        if self.daemon.poll() is None:self.daemon.kill()
        self.daemon.wait(timeout=10);self.daemon.stdout.close()

    def launch(self,mode):
        env={**os.environ,'XDG_STATE_HOME':str(self.folder/'xdg')}
        p=subprocess.Popen([sys.executable,str(self.folder/'server.py'),str(SOURCE),str(self.folder),mode],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
        self.servers.append(p);line=p.stdout.readline()
        self.assertTrue(line,p.stderr.read() if p.poll() is not None else 'no server reply');reply=json.loads(line)
        if 'child' in reply:self.native.append(reply['child'])
        return p,reply

    def wait_record(self,status='removed'):
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            records=list((self.folder/'state').glob('engine-*.json'))
            if records:
                row=json.loads(records[-1].read_text())
                if row['status']==status:return row
            time.sleep(.05)
        self.fail('lease did not reach '+status)

    def events(self):
        path=self.folder/'commands.jsonl'
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def test_registration_precedes_exec_and_restores_only_engine_eligibility(self):
        p,r=self.launch('normal');self.assertEqual(r['pid'],r['child']);self.assertEqual(r['affinity'],sorted(policy.effective_cpus()));self.assertEqual(len(r['parent_affinity']),1)
        event=self.events()[0];self.assertEqual((event['action'],event['pid']),('-x',r['pid']));self.assertLess(event['time'],float((self.folder/'executed').read_text()))
        p.stdin.write('quit\n');p.stdin.flush();p.wait(timeout=15);row=self.wait_record();self.assertEqual(row['engine_pid'],r['pid']);self.assertEqual([x['action'] for x in self.events()],['-x','-r'])

    def test_server_sigkill_still_removes_only_native_lease(self):
        p,r=self.launch('normal');p.kill();p.wait(timeout=10);row=self.wait_record();self.assertEqual(row['engine_pid'],r['pid']);self.assertEqual({x['pid'] for x in self.events()},{r['pid']})

    def test_native_sigkill_releases_lease_with_server_alive(self):
        p,r=self.launch('normal');os.kill(r['pid'],signal.SIGKILL);self.wait_record();self.assertIsNone(p.poll())

    def test_restart_has_a_new_pid_lease(self):
        p,first=self.launch('normal');p.stdin.write('restart\n');p.stdin.flush();second=json.loads(p.stdout.readline());self.native.append(second['child'])
        self.assertNotEqual(first['child'],second['child']);self.assertEqual(second['pid'],second['child'])
        events=self.events();self.assertEqual([(x['action'],x['pid']) for x in events],[('-x',first['child']),('-r',first['child']),('-x',second['child'])])
        p.stdin.write('quit\n');p.stdin.flush();p.wait(timeout=15)
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and len(self.events())<4:time.sleep(.05)
        self.assertEqual(self.events()[-1]['action'],'-r')

    def test_replacement_daemon_gets_the_existing_engine_lease(self):
        p,r=self.launch('normal');old=self.daemon;old.kill();old.wait(timeout=10);old.stdout.close()
        self.daemon=subprocess.Popen([sys.executable,'-c',"import ctypes,os,time;ctypes.CDLL(None).prctl(15,b'numad',0,0,0);print(os.getpid(),flush=True);time.sleep(600)"],stdout=subprocess.PIPE,text=True)
        new_pid=int(self.daemon.stdout.readline());(self.folder/'daemon.pid').write_text(str(new_pid))
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and len(self.events())<2:time.sleep(.05)
        self.assertEqual([(x['action'],x['pid']) for x in self.events()],[('-x',r['pid']),('-x',r['pid'])])
        p.stdin.write('quit\n');p.stdin.flush();p.wait(timeout=15);row=self.wait_record();self.assertEqual(row['daemon'][0],new_pid)

    def test_failed_exec_and_permission_denial_never_run_engine(self):
        for mode in ['exec-fail','deny']:
            with self.subTest(mode=mode):
                # Separate state per arm, preserving the previous failure record.
                if (self.folder/'state').exists():
                    (self.folder/'state').rename(self.folder/('state-before-'+mode))
                p,r=self.launch(mode);self.assertIn('error',r);p.wait(timeout=15);self.wait_record();self.assertFalse((self.folder/'executed').exists())

    def test_vision_and_explicit_child_cpu_allocations_are_not_widened(self):
        p,r=self.launch('vision');self.assertEqual(len(r['affinity']),1);self.assertFalse(self.events());p.stdin.write('quit\n');p.stdin.flush();p.wait(timeout=10)
        p,r=self.launch('explicit-mask');self.assertIn('error',r);p.wait(timeout=10);self.assertFalse(self.events())

    def test_absent_daemon_is_not_started_or_controlled(self):
        self.daemon.kill();self.daemon.wait(timeout=10);(self.folder/'daemon.pid').unlink()
        p,r=self.launch('normal');self.assertEqual(r['affinity'],sorted(policy.effective_cpus()));self.assertFalse(self.events())
        p.stdin.write('quit\n');p.stdin.flush();p.wait(timeout=10);self.wait_record('daemon-absent');self.assertFalse(self.events())


if __name__=='__main__':unittest.main()
