from sklearn.cluster import KMeans
global_player_team_dict = {}
class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = global_player_team_dict
        self.kMeans = None  # Ensure kMeans is initialized

    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kMeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kMeans.fit(image_2d)

        return kMeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get Clustering model
        kMeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kMeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kMeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kMeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kMeans.fit(player_colors)

        self.kMeans = kMeans

        self.team_colors[1] = kMeans.cluster_centers_[0]
        self.team_colors[2] = kMeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Get the team ID for a given player based on the player's color and the kMeans model.
        If the team ID is not found in player_team_dict, predict it using kMeans and store it.
        """
        # Check if player_id is already in the dictionary
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Check if kMeans model is initialized
        if self.kMeans is None:
            raise ValueError("kMeans model is not initialized. Please call assign_team_color() first.")

        # Extract player color
        player_color = self.get_player_color(frame, player_bbox)

        # Predict the team ID using kMeans
        team_id = self.kMeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        # Special case for player ID 91
        if player_id == 91:
            team_id = 1

        # Store the team ID in the global dictionary
        self.player_team_dict[player_id] = team_id
        return team_id


    def get_teams_by_player_ids(self, player_ids):
        """
        Get the team IDs for a list of player IDs based on a custom player_team_dict.
        Returns a dictionary where the keys are player IDs and the values are the corresponding team IDs.
        If a player ID is not found, the function will error.
        """
        team_ids = {}  # Initialize an empty dictionary
        
        # Ensure player_ids is iterable
        if not isinstance(player_ids, (list, tuple)):
            player_ids = [player_ids]
        for player_id in player_ids:
            if player_id in self.player_team_dict:
                team_ids[player_id] = self.player_team_dict[player_id]
            else:
                raise ValueError(f"Player ID {player_id} not found in player_team_dict.")

        return team_ids




