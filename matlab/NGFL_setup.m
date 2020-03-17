% add default path
function NGFL_setup()
disp('Running setup...');

[NGFL_dir,~,~] = fileparts(mfilename('fullpath'));
NGFL_conf.NGFL_dir = NGFL_dir;
disp(['novelGFL path: ' NGFL_dir]);

NGFL_conf.algorithm_path = fullfile(NGFL_dir,'algorithms');
NGFL_conf.libs_path = fullfile(NGFL_dir,'libs');
NGFL_conf.chicrime_path = fullfile(NGFL_dir,'chicago_crime_data_test');

% save(fullfile(NGFL_dir,'NGFL_conf.mat'),'NGFL_conf');

disp('Updating PATH!');

addpath(NGFL_conf.algorithm_path); % add algorithms into path in order to use them globally
addpath(NGFL_conf.libs_path); % add libraries into path in order to use them globally
addpath(NGFL_conf.chicrime_path); % add 'chicago_crime_data_test' folder into path in order to use them globally

disp('Setup finished!');
end