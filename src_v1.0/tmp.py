import pickle

rounds = ['basicICL_p0_n0','basicICL_hp0_n0', 'basicICL_p0_n10', 'basicICL_p0_hn10']
rounds = ['basicICL_p1_n0','basicICL_hp1_n0', 'basicICL_p1_n9', 'basicICL_p1_hn9']

for name in rounds:
    with open('out/pickles/report_'+name+'.pkl','rb') as f:
        x = pickle.load(f)

    a, p, n = [], [], []

    for _, (aa, pp, nn) in x.items():
        a.append(aa)
        p.append(pp)
        n.append(nn)

    fin_yes_acc = sum(p)/len(p)
    fin_no_acc = sum(n)/len(n)
    fin_acc = sum(a)/len(a)
    
    report_line = f"{name}, {fin_yes_acc}, {fin_no_acc}, {fin_acc}\n"
    with open('out/report_test.csv','a') as f:
        f.write(report_line)