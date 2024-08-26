import abc, os, csv

import gtimer as gt
from maple.core.rl_algorithm import BaseRLAlgorithm
from maple.data_management.replay_buffer import ReplayBuffer
from maple.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            eval_epoch_freq=1,
            expl_epoch_freq=1,
            eval_only=False,
            no_training=False,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            eval_epoch_freq=eval_epoch_freq,
            expl_epoch_freq=expl_epoch_freq,
            eval_only=eval_only,
            no_training=no_training,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        gt.reset_root()

        if self.trainer.plan_policy is not None:
            import datetime, os
            now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            model_dir = f"{self.trainer.env.task_name}/{now}/"
            self.path_dir = os.path.join("extra_model", model_dir)

            dirname = os.path.dirname(self.path_dir)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)

            if self.trainer.env.task_name == "Lift":
                self.waypoints_start = 24
            elif self.trainer.env.task_name == "PegInHole":
                self.waypoints_start = 27
            elif self.trainer.env.task_name == "Stack":
                self.waypoints_start = 34
            elif self.trainer.env.task_name == "NutAssemblyRound":
                self.waypoints_start = 25
            elif self.trainer.env.task_name == "Cleanup":
                self.waypoints_start = 25
            else:
                raise ValueError(f"Unknown task name: {self.trainer.env.task_name }")

    def _train(self):
        if self.min_num_steps_before_training > 0 and not self._eval_only:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=True, #False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs + 1),
                save_itrs=True,
        ):
            for pre_epoch_func in self.pre_epoch_funcs:
                pre_epoch_func(self, epoch)

            if epoch % self._eval_epoch_freq == 0:
                eval_paths = self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
                # header = ["end_effector_0", "end_effector_1", "end_effector_2", "way_point_0", "way_point_1", "way_point_2",]
                # for index in range(len(eval_paths)):
                #     csv_file, csv_logger = _get_csv_logger_(self.path_dir + f"/{epoch}_{index}")
                #     path = eval_paths[index]
                #     csv_logger.writerow(header)
                #     for i in range(path['observations'].shape[0]):
                #         data = [path['observations'][i][0].item(),
                #                 path['observations'][i][1].item(),
                #                 path['observations'][i][2].item(),
                #                 path['observations'][i][self.waypoints_start].item(),
                #                 path['observations'][i][self.waypoints_start+1].item(),
                #                 path['observations'][i][self.waypoints_start+2].item(),]
                #         csv_logger.writerow(data)
                #         csv_file.flush()

            gt.stamp('evaluation sampling')

            if not self._eval_only:
                for _ in range(self.num_train_loops_per_epoch):
                    if epoch % self._expl_epoch_freq == 0:
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            self.max_path_length,
                            self.num_expl_steps_per_train_loop,
                            discard_incomplete_paths=True, #False,
                        )
                        gt.stamp('exploration sampling', unique=False)

                        self.replay_buffer.add_paths(new_expl_paths)
                        gt.stamp('data storing', unique=False)

                    if not self._no_training:
                        self.training_mode(True)
                        for _ in range(self.num_trains_per_train_loop):
                            train_data = self.replay_buffer.random_batch(
                                self.batch_size)
                            self.trainer.train(train_data)
                        gt.stamp('training', unique=False)
                        self.training_mode(False)


            self._end_epoch(epoch)

def _get_csv_logger_(model_dir):
    csv_path = os.path.join(model_dir, f"log.csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
