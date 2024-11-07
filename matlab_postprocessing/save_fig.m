function save_fig(hfig,output_dir, file_name)

    figureName = [string(output_dir) + string(file_name)];

    % PDF figure:
    exportgraphics(hfig, figureName + ".pdf",'Resolution',600,'ContentType', 'vector') 

    % TIFF figure:
    exportgraphics(hfig, figureName + ".png",'Resolution',600) 

end