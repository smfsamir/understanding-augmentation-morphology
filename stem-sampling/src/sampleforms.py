from random import choices
import click

@click.command()
@click.option("--data_file", required=True)
@click.option("--res_file", required=True)
@click.option("--form_count", required=True)
def main(data_file, res_file, form_count):
    data_file = open(data_file)
    res_file = open(res_file, "w")
    form_count = int(form_count)

    paradigm = []
    for line in data_file:
        line = line.strip()
        if not line:
            if paradigm != []:
                paradigm = choices(paradigm, k=form_count)
                for line in paradigm:
                    print(line, file=res_file)
                print("",file=res_file)
            paradigm=[]
        else:
            paradigm.append(line)

if __name__=="__main__":
    main()
