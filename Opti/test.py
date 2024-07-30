¥documentclass{article}
¥usepackage{amsmath}
¥usepackage[dvipdfmx]{graphicx}
¥usepackage{here}
¥usepackage{listings}

¥begin{document}
サーバのリソースが一定である時の、システムの可用性が最大となるサービスの実装形態や各ソフトウェアの冗長度合いを探る

¥section*{システムモデル}
¥begin{itemize}
    ¥item ¥textbf{サービス ($M$個)}:
    ¥begin{itemize}
        ¥item サービスはアプリケーションシステムを実現するために必要となる、部分的な機能を提供する。$i$番目のサービスを$m_i$と表す
        ¥item サービス$m_i$の可用性を $a_{m_i}$ とする
    ¥end{itemize}
    
    ¥item ¥textbf{サーバ ($H$)}:
    ¥begin{itemize}
        ¥item サーバは複数のソフトウェアをホスト可能。$j$台目のサーバを$h_j$と表す
        ¥item 全てのサーバの可用性は等しいものとし $a_{h}$ とする
        ¥item 各サーバに1つのソフトウェアがホストされるものとする
        ¥item サーバ $j$にソフトウェア $k$がホストされている状態を$b_{(j,k)}$という変数を用いて$b_{(j,k)}=1$と表し、そうでなければ$0$とする
    ¥end{itemize}
     
    ¥item ¥textbf{ソフトウェア ($S$個)}:
    ¥begin{itemize}
        ¥item サービスをサーバへ実装する際の実装単位。$k$個目のソフトウェアを$s_k$と表す
        ¥item 各サービスは1つのソフトウェアにのみ内包されており、ソフトウェア$k$にサービス $i$が内包されている状態を
        $c_{(k,i)}$という変数を用いて$c_{(k,i)}=1$と表し、そうでなければ$0$とする.¥¥
        $s_k$に含まれるサービスの数を$s_{k_{m}}$とする。$s_{k_{m}}$は下のように表される
        ¥[s_{k_{m}} = ¥sum_{i}c_{k,i}¥]
        ¥item ソフトウェア$s_k$の可用性を $a_{s_k}$ とし、ソフトウェア$s_k$が内包するサービスの可用性の積に等しいものとする。$a_{s_k}$は下のように表される
	¥[a_{s_{k}} = ¥prod_{i | c_{k,i}=1}a_{m_{i}}¥]¥
	 ¥item ソフトウェアは複数のサーバに冗長化して配置可能
	¥item ソフトウェア$s_k$がホストされているサーバ台数 $r_k$ は下のように表される
		¥[r_k = ¥sum_{j}b_{j,k}¥]
  また、ソフトウェア$s_k$が必要とするリソース$R_k$は、モノリシックやマイクロサービスへのなりやすさ$r_{add}$を用いて以下のように表される。
  ¥[R_{k} = r_{k} ¥cdot ¥{1+(s_{k_{m}}-1) ¥cdot r_{add}¥} ¥]¥
	¥item ソフトウェアがシステム内で少なくとも1つ動いていれば、そのソフトウェアは可用であるとする。そのため、システムにおけるソフトウェア$s_k$ の可用性は下のように表される
	¥[1-(1-a_{s_k}¥cdot a_h)^{r_k}¥] 
	¥end{itemize}
	
	¥item ¥textbf{アプリケーションシステム}:¥¥
	全てのサービスが実行されることでアプリケーションシステムは可用となる。そのため、アプリケーションシステムの可用性$a_{sys}$は下のように表される
	¥[
	a_{sys} = ¥prod_{k}{(1-(1-a_{s_{k}})^{r_k})}
	¥]
    
¥end{itemize}
     
¥section*{ソフトウェアの冗長化度やサービスの実装形態の最適化}
¥begin{itemize}
        ¥item¥textbf{決定変数}:
        $c_{k,i},b_{j,k}$
        
        ¥item¥textbf{目的関数}:
        max $a_{sys}$
        
        ¥item¥textbf{制約条件}:
        システムのリソースを$R$で表す時、制約条件は以下の通り
        ¥begin{itemize}
        ¥item システムにおいて使用される合計リソースは$R$以下
       	 	¥[¥sum_{k}R_k ¥leq R ¥]
        ¥item 各サービスが1つのソフトウェアにのみ内包されている
		¥[¥sum_{k}c_{k,i} = 1¥quad,¥forall i ¥]  
	¥item 各サーバに1つのソフトウェアがホストされている	
		 ¥[¥sum_{k} b_{j,k} = 1 ¥quad, ¥forall j ¥]
    ¥end{itemize}
¥end{itemize}

¥section*{解析的評価}
システムの可用性が最大となる, ソフトウェアの数($S$),サービス配置($c_{k,i}$),ソフトウェアごとの冗長化数($r_k$)を明らかにする。
10個のサービスを含むシステムにおいて、ソフトウェア数の変化、サービス実装形態、冗長化度合いがそれぞれ、どれほどの影響をシステム可用性に与えているのか、リソースや$r_{add}$を変化させて解析を行った。

解析においては、それぞれの影響を見るために下の3つの分布をプロットした。
¥begin{itemize}
	¥item いろいろな実装形態に対して、冗長化パターンとソフトウェア数を全て探索して最適化し、その可用性の分布を見る
	¥item いろいろな冗長化パターンに対して、実装形態とソフトウェア数を全て探索して最適化し、その可用性の分布を見る
	¥item いろいろなソフトウェア数に対して、冗長化パターンと実装形態を全て探索して最適化し、その可用性の分布を見る
¥end{itemize}

なお、画像におけるグラフにおいて、横軸がシステム可用性、縦軸が累積密度関数となっており、ある要素に関するプロットが、様々なシステム可用性をとるほど（プロットされた線の横幅が大きいほど）その要素がシステム可用性に与える影響が大きい。以下の全てのグラフにおいて、サービス可用性はすべて0.99であり、サーバ可用性も0.99である。
¥begin{figure}[H]
¥centering  % 図を真ん中に配置
¥includegraphics[width = 16cm]{./image/comparison.png}
¥caption{要素ごとの影響の比較}    ¥label{comparison}
¥end{figure}

図¥ref{comparison}より以下のことがわかる
¥begin{itemize}
¥item $r_{add}$が小さく、リソースが十分にある時、冗長化度合いが大きな影響を与えている。これは、システムに含まれるソフトウェア全てが十分に冗長化できる場合、サービスの実装形態やソフトウェアはシステム可用性にあまり影響を与えないためである。
¥item リソースが少ないか、$r_{add}$が大きい時、冗長化度が一番大きな影響を与えているが、サービスの実装形態やソフトウェア数も影響を与えている。これは、リソースが少なく十分な冗長化が行えない場合、サービスの実装形態やソフトウェア数も重要になってくるためである。
¥end{itemize}

また、ソフトウェア数とサービス実装形態が同じような影響を及ぼす理由として、


次に、サービス可用性やサーバ可用性を変化させたときの、それぞれの要素が及ぼす影響の変化を見ました。
サーバ可用性を変化させた際の比較（実線はサーバ可用性が0.99、破線はサーバ可用性が0.95）
¥begin{figure}[H]
¥centering  % 図を真ん中に配置
¥includegraphics[width = 16cm]{./image/various_server_avail.png}
¥caption{サーバ可用性を変化させた際の比較}    ¥label{server_avail}
¥end{figure}

図¥ref{server_avail}より以下のことがわかる
¥begin{itemize}
¥item サーバ可用性を低下させることで、すべての要素が与える影響は大きくなった。これは、サーバ可用性の低下により、システム可用性が大きく低下する組み合わせが存在するためである。（例:ソフトウェア数が10個でそれぞれにサービスが1つづつ実装されている時）
¥item リソースが少なくなることで、サービス実装形態・冗長化度合い・ソフトウェア数それぞれの影響が大きくなる。これは、リソースが少ないことにより、可用性が小さくなるような組み合わせが十分に冗長化されなかったり、可用性が高くなるようなサービス実装形態を取れなくなるためである。
¥item $r_{add}$が大きくなることで、ソフトウェア数とサービス実装形態の影響が大きくなった。これは、$r_{add}$が大きいと、1つのソフトウェアに多くのサービスを含む場合はリソースを多く必要とするため冗長化が十分に行えず、細かくサービスを分けるとサーバ可用性の影響が大きくなるなど、可用性の低くなる組み合わせが生じるためである。
¥end{itemize}


¥begin{figure}[H]
¥centering  % 図を真ん中に配置
¥includegraphics[width = 16cm]{./image/various_service_avail.png}
¥caption{サービス可用性を変化させた際の比較}    ¥label{service_avail}
¥end{figure}
図¥ref{service_avail}より以下のことがわかる
¥begin{itemize}
¥item リソースが減少することで、ソフトウェア数・サービス実装形態・冗長化度合いの影響が大きくなった。これは、
¥item ${r_{add}}$が増加することでソフトウェア数とサービス実装形態の及ぼす影響が大きくなった。
¥item サービス可用性が一部低下することで、すべての影響が増加する。
¥end{itemize}

また、それぞれのリソース・$r_{add}$において、最も高いシステム可用性を示すときの組み合わせは以下のようになる。
システムが最大となる組み合わせについて、リソースが少ない時は、1つのソフトウェアに多くのサービスが含まれることが多いが、リソースが多くなるにつれ、$r_{add}$の影響が現れ、ソフトウェアに含まれるサービス数が小さくなる。
リソースが小さいと、$r_{add}$の影響があまり表れない。これは、冗長化度合の影響がサービスの実装形態よりも大きいため、なるべく冗長化が起きるよう、1つのソフトウェアのサービス数を少なくしているためである。




¥end{document}