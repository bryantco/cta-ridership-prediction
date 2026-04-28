all:
  cd extract_ridership_data && just
  cd feature_engineer && just
  cd predict_ridership && just
  cd evaluate && just

clean:
  cd extract_ridership_data && just clean
  cd feature_engineer && just clean
  cd predict_ridership && just clean
  cd evaluate && just clean

