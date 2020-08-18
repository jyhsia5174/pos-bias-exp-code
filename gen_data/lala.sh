#! /bin/zsh

cd split-data
git add -f context.ffm item.ffm
cd ..
cd ocffm-to-ocsvm
cd ..
cd grid-and-filter
cd hyffm-grid
git add -f item.ffm rd.tr.ffm rd.trva.ffm rd.va.ffm 
cd ..
cd filter
git add -f filter.ffm item.ffm  rd.trva.ffm filter.model
cd ..
cd ..
cd add_bias
git add -f det.ffm prop.ffm random.ffm 
cd ..
cd ocffm-to-ocsvm
git add -f det.ffm.pos.0.5.bias prop.ffm.pos.0.5.bias random.ffm.pos.0.5.bias item.ffm truth.ffm 
cd ..
cd add_bias
cd ..
cd ocffm-to-ocsvm
rm det.ffm.pos.0.5.bias prop.ffm.pos.0.5.bias random.ffm.pos.0.5.bias 
ln -s ../add_bias/*.bias .
git add -f det.ffm.pos.0.5.bias prop.ffm.pos.0.5.bias random.ffm.pos.0.5.bias item.ffm truth.ffm 
cd ..
cd kkbox_csv_to_ocffm
git add -f context_csv_to_ffm.py.ipynb filter-by-artist.py.ipynb item_csv_to_ffm.py.ipynb 
#gc -m "Track symbolic link and ipynb"
