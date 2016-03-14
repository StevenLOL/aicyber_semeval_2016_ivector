import sys

def main(fin,fout):
    print 'fin',fin
    print 'fout',fout
    assert fin!=fout
    fin=open(fin).readlines()
    fout=open(fout,'w')
    for f in fin:
        #print f
        uid=f.split()[0]
        fout.write('%s %s\n'%(uid,uid))
    fout.close()


if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])