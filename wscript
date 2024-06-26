#!/usr/bin/env python
# encoding: utf-8
import sys
import os
import fnmatch
import glob
sys.path.insert(0, sys.path[0]+'/waf_tools')

VERSION = '1.0.0'
APPNAME = 'simple_sim'

srcdir = '.'
blddir = 'build'

from waflib.Build import BuildContext
from waflib import Logs
from waflib.Tools import waf_unit_test


def options(opt):
    opt.load('compiler_cxx')
    opt.load('compiler_c')
    opt.load('eigen')
    opt.load('proxqp')
    opt.load('simde')

    opt.add_option('--no-native', action='store_true', help='Do not compile with march=native optimization flags', dest='no_native')

def configure(conf):
    conf.load('compiler_cxx')
    conf.load('compiler_c')
    conf.load('eigen')
    conf.load('proxqp')
    conf.load('simde')

    conf.check_eigen(required=True)
    conf.check_proxqp(required=True)
    conf.check_simde(required=False)

    native = '-march=native -DPROXSUITE_VECTORIZE'
    native_icc = 'mtune=native -DPROXSUITE_VECTORIZE'

    if conf.options.no_native:
        native = ''
        native_icc = ''
    elif 'INCLUDES_SIMDE' not in conf.env:
        native = '-march=native'
        native_icc = 'mtune=native'

    # We require C++17
    if conf.env.CXX_NAME in ["icc", "icpc"]:
        common_flags = "-Wall -std=c++17"
        opt_flags = " -O3 -xHost -unroll -g " + native_icc
    elif conf.env.CXX_NAME in ["clang"]:
        common_flags = "-Wall -std=c++17"
        # no-stack-check required for Catalina
        opt_flags = " -O3 -g -faligned-new -fno-stack-check -Wno-narrowing" + native
    else:
        gcc_version = int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1])
        if gcc_version < 50:
            conf.fatal('We need C++17 features. Your compiler does not support them!')
        else:
            common_flags = "-Wall -std=c++17"
        opt_flags = " -O3 -g " + native
        if gcc_version >= 71:
            opt_flags = opt_flags + " -faligned-new"

    all_flags = common_flags + opt_flags + " -DNDEBUG"
    conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + all_flags.split()
    print('C++ flags:', conf.env['CXXFLAGS'])

def build(bld):
    libs_qp = 'EIGEN PROXQP SIMDE '

    lib_files = 'src/srbd/srbd.cpp'

    bld.program(features = 'cxx cxxstlib',
                source = lib_files,
                includes = './src',
                uselib = libs_qp,
                target = 'srbd')

    bld.program(features = 'cxx',
                install_path = None,
                source = 'src/examples/loco.cpp',
                includes = './src',
                uselib = libs_qp,
                use = 'srbd',
                target = 'loco')

    # Install headers
    install_files = []
    for root, dirnames, filenames in os.walk(bld.path.abspath()+'/src/'):
        for filename in fnmatch.filter(filenames, '*.hpp'):
            install_files.append(os.path.join(root, filename))
    install_files = [f[len(bld.path.abspath())+1:] for f in install_files]

    for f in install_files:
        end_index = f.rfind('/')
        if end_index == -1:
            end_index = len(f)
        bld.install_files('${PREFIX}/include/' + f[4:end_index], f)

    # Install library
    bld.install_files('${PREFIX}/lib', blddir + '/libsrbd.a')