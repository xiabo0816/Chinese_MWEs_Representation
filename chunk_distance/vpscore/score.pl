#!/usr/bin/perl 
#  @ARGV[0] inputfile
#  @ARGV[1] outputfile
#  @ARGV[2] maxnverb
#  @ARGV[3] type
#  @ARGV[4] topN
#  http://localhost:9091/chunk_distance?query=他们 对 历史 进行 反思&head=2
#D:\yuliao\sentence>perl score.pl test k5 1 2 1

use LWP;
use utf8;
use Encode;

$G_MAXVERBCOUNT = 2;        # 是否仅多动词句子的开关

if(@ARGV < 0){
    print "Usage: <inputfile> <outputfile> <maxnverb> <type> <topN>\n";
}else{
    $G_MAXVERBCOUNT = @ARGV[2];

    # 只读方式打开文件
    open(DATA1, "<".@ARGV[0]);
    
    # 打开新文件并写入
    open(DATA2, ">".@ARGV[1]);

    while(<DATA1>)
    {
        chomp;
        @personal = split(/_/, $_);
        if(@personal < 4) {next;}
        @withPos = split(/ /, @personal[0]);
        @withoutPos = split(/ /, @personal[1]);
        $nVCount = 0;
        $t = -1;
        foreach(@withPos){
            $t++;
            @pos = split(/\//, @withPos[$t]);
            if((@pos[1] eq decode('utf8', 'v')) || (@pos[1] eq decode('utf8', 'vn'))) {$nVCount++;}
        }
        if($nVCount<=$G_MAXVERBCOUNT) {next;}

        $i = -1;
        $t = -1;
        foreach(@withoutPos){  
            $t++;
            if($_ eq $personal[3]){  
                $i = $t;
                last;
            }  
        }
        if($i == -1){ next; }

        # print DATA2 @personal[2];
        my $browser = LWP::UserAgent->new();
        my $seite = $browser->get('http://localhost:9092/chunk_distance?type='.@ARGV[3].'&query='.encode("gbk", decode("utf8", @personal[1])).'&head='.$i);
        my $seite_code = $seite->decoded_content();
        if($seite->is_success) {
            if(!length($seite_code)){ next; }
            $seite_code = encode("utf8", decode("gbk", $seite_code));
            print DATA2 @personal;
            print DATA2 "\n";
            print DATA2 $seite_code;
            $ntotal++;
            @result = split(/\n/, $seite_code);
            $i = -1;
            $top = @ARGV[4];
            foreach(@result)  
            {  
                $i++;
                if($i > $top) {last;}
                
                @key = split(/\t/, @result[$i]);

                $p = -1;
                $t = -1;
                foreach(@withoutPos){  
                    $t++;
                    if($_ eq @key[0]){  
                        $p = $t;
                        last;
                    }  
                }
                @pos = split(/\//, @withPos[$p]);
                if((@pos[1] eq "v")) {
                    if(@key[0] eq @personal[2])
                    {  
                        $nCorrect++;
                        print DATA2 "√-------------\n";
                        print DATA2 "\n";
                    }
                }else{
                    $top++;
                }
            } 
        }
    }
    print "nCorrect = $nCorrect\n ntotal = $ntotal\n nCorrect/ntotal = ".$nCorrect/$ntotal;
    close( DATA1 );
    close( DATA2 );
}