function ImportDataSingle()

    % dataTable = readtable('vt_test_all.xlsx','Sheet','Sheet1'); %验证完整运行线的能耗情况
    dataTable = readtable('vt_data.csv'); 
    TimeList = dataTable{:, 1};
    Velocity_List = dataTable{:, 2};
    mess_1 = dataTable{1:26, 3};
    mess_2 = dataTable{1:26, 4};
    mess_3 = dataTable{1:26, 5};
    mess_4 = dataTable{1:26, 6};
    mess_5 = dataTable{1:26, 7};
    mess_6 = dataTable{1:26, 8};

    assignin('base','Velocity_Series',timeseries(Velocity_List,TimeList));
    assignin('base','mess_1',dataTable{1:26, 3}');
    assignin('base','mess_2',dataTable{1:26, 4}');
    assignin('base','mess_3',dataTable{1:26, 5}');
    assignin('base','mess_4',dataTable{1:26, 6}');
    assignin('base','mess_5',dataTable{1:26, 7}');
    assignin('base','mess_6',dataTable{1:26, 8}');

end