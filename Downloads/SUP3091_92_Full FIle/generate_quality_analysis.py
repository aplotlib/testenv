import matplotlib
matplotlib.use('Agg')

from quality.page1_defect_scope  import build_page as build_p1
from quality.page2_early_signals import build_page as build_p2
from quality.page3_cost_of_delay import build_page as build_p3
from quality.page4_current_state import build_page as build_p4
from quality.canvas import save_all


def main():
    pages = [build_p1(), build_p2(), build_p3(), build_p4()]
    pdf_path = save_all(pages)
    print(f"Saved PDF : {pdf_path}")
    for i in range(1, 5):
        print(f"  Page {i}  : {pdf_path.replace('.pdf', f'_p{i}.png')}")


if __name__ == '__main__':
    main()
